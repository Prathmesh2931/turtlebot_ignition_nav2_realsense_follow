#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import sounddevice as sd
import soundfile as sf
import subprocess
from threading import Thread
from dotenv import load_dotenv
from groq import Groq
import re
import numpy as np
from pathlib import Path

# Dynamically locate the .env file — works in both src/ and install/ modes
from pathlib import Path
from dotenv import load_dotenv


class VoiceObjectController(Node):
    def __init__(self):
        super().__init__('voice_object_controller')

        # Load environment variables

        # Find .env file in package source directory (even when installed)
        # dotenv_path = Path(__file__).resolve().parents[2] / 'src' / 'navigation2_ignition_gazebo_turtlebot3' / 'object_tracking' / 'src' / '.env'
        # load_dotenv(dotenv_path)
        current_dir = Path(__file__).resolve()
        for parent in current_dir.parents:
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                break
        else:
            print("⚠️  .env file not found anywhere above this script.")


        # Get Groq API key securely
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            self.get_logger().error("❌ GROQ_API_KEY not found in environment! Check your .env file.")
            raise ValueError("Missing GROQ_API_KEY in .env")

        # Initialize Groq client
        self.client = Groq(api_key=api_key)

        
        # Groq API setup with updated model
        # api_key = os.environ.get("GROQ_API_KEY", "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")
        # self.client = Groq(api_key=api_key)
        
        # Publishers
        self.target_object_pub = self.create_publisher(String, '/target_object', 10)
        self.control_mode_pub = self.create_publisher(String, '/control_mode', 10)
        self.search_mode_pub = self.create_publisher(Bool, '/search_mode', 10)
        
        # Simple command mappings for direct parsing (fallback when LLM fails)
        self.direct_object_commands = {
            # Follow commands
            "follow person": "person",
            "track person": "person",
            "find person": "person",
            "follow human": "person",
            "track human": "person",
            "find human": "person",
            
            "follow chair": "chair",
            "track chair": "chair", 
            "find chair": "chair",
            
            "follow bottle": "bottle",
            "track bottle": "bottle",
            "find bottle": "bottle",
            
            "follow laptop": "laptop",
            "track laptop": "laptop",
            "find laptop": "laptop",
            
            "follow phone": "cell phone",
            "track phone": "cell phone",
            "find phone": "cell phone",
            
            "follow car": "car",
            "track car": "car",
            "find car": "car",
            
            "follow dog": "dog",
            "track dog": "dog",
            "find dog": "dog",
            
            "follow cat": "cat",
            "track cat": "cat",
            "find cat": "cat",
            
            "follow cup": "cup",
            "track cup": "cup",
            "find cup": "cup",
            
            "follow book": "book",
            "track book": "book",
            "find book": "book",
            
            "follow bag": "backpack",
            "track bag": "backpack",
            "find bag": "backpack",
            
            "follow backpack": "backpack",
            "track backpack": "backpack",
            "find backpack": "backpack",
            
            # Color objects
            "follow red box": "red_object",
            "track red box": "red_object",
            "find red box": "red_object",
            "follow red object": "red_object",
            "track red object": "red_object",
            "find red object": "red_object",
            
            "follow blue box": "blue_object",
            "track blue box": "blue_object", 
            "find blue box": "blue_object",
            
            "follow green box": "green_object",
            "track green box": "green_object",
            "find green box": "green_object",
        }
        
        # Control commands
        self.control_commands = {
            "stop": "stop",
            "halt": "stop", 
            "pause": "pause",
            "resume": "resume",
            "search": "search",
            "look around": "search",
            "find something": "search",
            "scan area": "search",
        }
        
        # Single word object mappings
        self.single_word_objects = {
            "person": "person",
            "human": "person",
            "chair": "chair",
            "bottle": "bottle",
            "laptop": "laptop",
            "phone": "cell phone",
            "car": "car",
            "dog": "dog",
            "cat": "cat",
            "cup": "cup",
            "book": "book",
            "bag": "backpack",
            "backpack": "backpack",
        }
        
        self.get_logger().info("Voice Object Controller initialized")
        self.get_logger().info("Available commands: follow/track/find [object], stop, search")
        
        # Start voice recognition loop
        self.start_voice_recognition()
    
    def start_voice_recognition(self):
        """Start continuous voice recognition in a separate thread"""
        self.voice_thread = Thread(target=self.voice_recognition_loop, daemon=True)
        self.voice_thread.start()
    
    def voice_recognition_loop(self):
        """Continuous voice recognition loop"""
        try:
            while rclpy.ok():
                try:
                    # Record audio
                    audio_file = self.record_audio(duration=4)
                    
                    # Transcribe audio
                    transcript = self.transcribe_audio(audio_file)
                    transcript = self.clean_text(transcript)
                    
                    if transcript and len(transcript.strip()) > 0:
                        self.get_logger().info(f"Heard: '{transcript}'")
                        
                        # Parse and execute command
                        self.parse_and_execute_command(transcript)
                    
                except Exception as e:
                    self.get_logger().error(f"Voice recognition error: {e}")
                    
        except KeyboardInterrupt:
            self.get_logger().info("Voice recognition stopped")
    
    def record_audio(self, duration=4, sample_rate=16000, output_file="/tmp/voice_input.wav"):
        """Record audio from microphone"""
        try:
            self.get_logger().debug("Listening...")
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
            sd.wait()
            
            # Check if audio has meaningful content
            if np.max(np.abs(audio)) < 500:  # Very quiet audio threshold
                return None
                
            sf.write(output_file, audio, sample_rate)
            return output_file
        except Exception as e:
            self.get_logger().error(f"Audio recording error: {e}")
            return None
    
    def transcribe_audio(self, file_path):
        """Transcribe audio using Groq Whisper"""
        if not file_path:
            return ""
            
        try:
            with open(file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=(file_path, audio_file.read()),
                    model="whisper-large-v3-turbo",
                    language="en",
                )
            return transcription.text
        except Exception as e:
            self.get_logger().error(f"Transcription error: {e}")
            return ""
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        return text.lower().strip().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
    
    def parse_and_execute_command(self, command):
        """Parse voice command with fallback mechanisms"""
        # First try direct pattern matching (fastest and most reliable)
        if self.try_direct_pattern_matching(command):
            return
        
        # Then try LLM parsing as fallback
        try:
            self.try_llm_parsing(command)
        except Exception as e:
            self.get_logger().error(f"LLM parsing failed: {e}")
            # Final fallback: try simple keyword matching
            self.try_keyword_matching(command)
    
    def try_direct_pattern_matching(self, command):
        """Try direct pattern matching first"""
        # Check exact matches in command dictionaries
        if command in self.direct_object_commands:
            object_name = self.direct_object_commands[command]
            self.execute_object_tracking(object_name)
            return True
        
        if command in self.control_commands:
            control_action = self.control_commands[command]
            self.execute_control_command(control_action)
            return True
        
        # Check for pattern: "follow/track/find [object]"
        for action in ["follow", "track", "find"]:
            if command.startswith(action + " "):
                object_part = command.replace(action + " ", "").strip()
                if object_part in self.single_word_objects:
                    object_name = self.single_word_objects[object_part]
                    self.execute_object_tracking(object_name)
                    return True
                # Check for color objects
                color_command = action + " " + object_part
                if color_command in self.direct_object_commands:
                    object_name = self.direct_object_commands[color_command]
                    if "object" in object_name:  # It's a color object
                        self.execute_color_object_tracking(object_name)
                    else:
                        self.execute_object_tracking(object_name)
                    return True
        
        return False
    
    def try_llm_parsing(self, command):
        """Try LLM parsing with updated model"""
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a command parser for robot object tracking. 
                    
                    Available object commands: {list(self.direct_object_commands.keys())}
                    Available control commands: {list(self.control_commands.keys())}
                    
                    Parse the user's command and return ONLY ONE of these formats:
                    - "OBJECT:[object_name]" for tracking commands (e.g., "OBJECT:person")
                    - "CONTROL:[action]" for control commands (e.g., "CONTROL:stop") 
                    - "COLOR_OBJECT:[color]_object" for colored objects (e.g., "COLOR_OBJECT:red_object")
                    - "INVALID" if command doesn't match
                    
                    Examples:
                    - "follow the person" -> "OBJECT:person"
                    - "track a chair" -> "OBJECT:chair"
                    - "find red box" -> "COLOR_OBJECT:red_object"
                    - "stop robot" -> "CONTROL:stop"
                    - "search around" -> "CONTROL:search"
                    """
                },
                {
                    "role": "user",
                    "content": f"Parse this command: '{command}'"
                }
            ],
            model="llama-3.1-8b-instant",  # Updated to supported model
        )
        
        parsed_result = chat_completion.choices[0].message.content.strip().strip('"').strip("'")
        self.get_logger().info(f"LLM parsed command: {parsed_result}")
        
        # Execute based on parsed result
        if parsed_result.startswith("OBJECT:"):
            object_name = parsed_result.replace("OBJECT:", "")
            self.execute_object_tracking(object_name)
            
        elif parsed_result.startswith("COLOR_OBJECT:"):
            color_object = parsed_result.replace("COLOR_OBJECT:", "")
            self.execute_color_object_tracking(color_object)
            
        elif parsed_result.startswith("CONTROL:"):
            control_action = parsed_result.replace("CONTROL:", "")
            self.execute_control_command(control_action)
            
        else:
            self.get_logger().warn(f"LLM could not understand command: '{command}'")
            raise Exception("LLM parsing failed")
    
    def try_keyword_matching(self, command):
        """Final fallback: simple keyword matching"""
        self.get_logger().info("Trying keyword matching as final fallback")
        
        # Check for control keywords
        if any(word in command for word in ["stop", "halt"]):
            self.execute_control_command("stop")
            return
        
        if any(word in command for word in ["search", "look", "scan"]):
            self.execute_control_command("search")
            return
        
        if "pause" in command:
            self.execute_control_command("pause")
            return
        
        if "resume" in command:
            self.execute_control_command("resume")
            return
        
        # Check for object keywords
        for obj_key, obj_value in self.single_word_objects.items():
            if obj_key in command:
                if any(action in command for action in ["follow", "track", "find"]):
                    self.execute_object_tracking(obj_value)
                    return
        
        # Check for color objects
        if "red" in command and ("box" in command or "object" in command):
            self.execute_color_object_tracking("red_object")
            return
        
        if "blue" in command and ("box" in command or "object" in command):
            self.execute_color_object_tracking("blue_object")
            return
        
        if "green" in command and ("box" in command or "object" in command):
            self.execute_color_object_tracking("green_object")
            return
        
        self.get_logger().warn(f"Could not understand command with any method: '{command}'")
    
    def execute_object_tracking(self, object_name):
        """Execute YOLO-based object tracking"""
        self.get_logger().info(f"Starting to track: {object_name}")
        
        # Publish target object
        target_msg = String()
        target_msg.data = object_name
        self.target_object_pub.publish(target_msg)
        
        # Set control mode to following
        mode_msg = String()
        mode_msg.data = "following"
        self.control_mode_pub.publish(mode_msg)
        
        # Disable search mode
        search_msg = Bool()
        search_msg.data = False
        self.search_mode_pub.publish(search_msg)
    
    def execute_color_object_tracking(self, color_object):
        """Execute color-based object tracking"""
        self.get_logger().info(f"Starting to track: {color_object}")
        
        # Publish color object target
        target_msg = String()
        target_msg.data = color_object
        self.target_object_pub.publish(target_msg)
        
        # Set control mode to color tracking
        mode_msg = String()
        mode_msg.data = "color_tracking"
        self.control_mode_pub.publish(mode_msg)
        
        # Disable search mode
        search_msg = Bool()
        search_msg.data = False
        self.search_mode_pub.publish(search_msg)
    
    def execute_control_command(self, action):
        """Execute control commands"""
        self.get_logger().info(f"Executing control action: {action}")
        
        if action in ["stop", "halt"]:
            # Stop all tracking
            mode_msg = String()
            mode_msg.data = "stopped"
            self.control_mode_pub.publish(mode_msg)
            
        elif action == "pause":
            mode_msg = String()
            mode_msg.data = "paused"
            self.control_mode_pub.publish(mode_msg)
            
        elif action == "resume":
            mode_msg = String()
            mode_msg.data = "following"
            self.control_mode_pub.publish(mode_msg)
            
        elif action == "search":
            # Enable search mode
            search_msg = Bool()
            search_msg.data = True
            self.search_mode_pub.publish(search_msg)
            
            mode_msg = String()
            mode_msg.data = "searching"
            self.control_mode_pub.publish(mode_msg)

def main(args=None):
    rclpy.init(args=args)
    
    node = VoiceObjectController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()