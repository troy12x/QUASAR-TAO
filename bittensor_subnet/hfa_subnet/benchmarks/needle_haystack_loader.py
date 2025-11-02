# The MIT License (MIT)
# Copyright © 2024 HFA Research Team

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import random
import os
from typing import List, Dict, Any, Optional
from .benchmark_task import BenchmarkTask

import bittensor as bt


class NeedleHaystackLoader:
    """
    Loader for Needle-in-Haystack evaluation tasks.
    
    The Needle-in-Haystack test evaluates a model's ability to retrieve specific
    information (the "needle") from within a large context (the "haystack").
    This is a critical test for infinite context capabilities as it directly
    measures whether models can maintain perfect recall across long sequences.
    
    This loader generates synthetic needle-in-haystack tasks with:
    - Configurable context lengths (1K to 100K+ tokens)
    - Multiple needle types (facts, numbers, names, dates)
    - Variable needle positions (beginning, middle, end)
    - Distractor content to increase difficulty
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Needle-in-Haystack loader.
        
        Args:
            config: Configuration dictionary with needle-haystack settings
        """
        self.config = config
        self.min_context_length = config.get('min_context_length', 1000)
        self.max_context_length = config.get('max_context_length', 100000)
        self.needle_types = config.get('needle_types', [
            'fact', 'number', 'name', 'date', 'location', 'quote'
        ])
        self.position_types = config.get('position_types', [
            'beginning', 'early', 'middle', 'late', 'end'
        ])
        
        # Content sources for generating haystack
        self.haystack_topics = config.get('haystack_topics', [
            'technology', 'science', 'history', 'literature', 'economics',
            'politics', 'environment', 'culture', 'sports', 'medicine'
        ])
        
        # Needle templates for different types
        self.needle_templates = {
            'fact': [
                "The important fact to remember is that {value}.",
                "It should be noted that {value}.",
                "A key point is that {value}.",
                "Remember this crucial information: {value}."
            ],
            'number': [
                "The magic number is {value}.",
                "The secret code is {value}.",
                "The important number to remember is {value}.",
                "The key figure is {value}."
            ],
            'name': [
                "The person you need to remember is {value}.",
                "The key individual is {value}.",
                "Remember this name: {value}.",
                "The important person is {value}."
            ],
            'date': [
                "The crucial date is {value}.",
                "Remember this important date: {value}.",
                "The key date to note is {value}.",
                "The significant date is {value}."
            ],
            'location': [
                "The important location is {value}.",
                "The key place to remember is {value}.",
                "The crucial location is {value}.",
                "Remember this place: {value}."
            ],
            'quote': [
                "The important quote is: '{value}'",
                "Remember this key quote: '{value}'",
                "The crucial statement is: '{value}'",
                "The significant quote is: '{value}'"
            ]
        }
        
        bt.logging.info(f"NeedleHaystackLoader initialized with {len(self.needle_types)} needle types")
    
    def is_available(self) -> bool:
        """Needle-in-haystack is always available as it generates synthetic tasks"""
        return True
    
    def load_tasks(self, 
                   num_tasks: int,
                   context_length_range: Optional[tuple] = None) -> List[BenchmarkTask]:
        """
        Generate Needle-in-Haystack tasks.
        
        Args:
            num_tasks: Number of tasks to generate
            context_length_range: Optional tuple of (min_length, max_length) for context
            
        Returns:
            List of BenchmarkTask objects
        """
        tasks = []
        
        for i in range(num_tasks):
            try:
                task = self._generate_needle_task(i, context_length_range)
                if task:
                    tasks.append(task)
            except Exception as e:
                bt.logging.error(f"Error generating needle-haystack task {i}: {e}")
        
        bt.logging.info(f"Generated {len(tasks)} needle-in-haystack tasks")
        return tasks
    
    def _generate_needle_task(self, 
                            task_id: int,
                            context_length_range: Optional[tuple] = None) -> Optional[BenchmarkTask]:
        """Generate a single needle-in-haystack task"""
        
        try:
            # Determine context length
            if context_length_range:
                min_len, max_len = context_length_range
            else:
                min_len, max_len = self.min_context_length, self.max_context_length
            
            target_length = random.randint(min_len, max_len)
            
            # Select needle type and position
            needle_type = random.choice(self.needle_types)
            position_type = random.choice(self.position_types)
            
            # Generate needle content
            needle_value, needle_text = self._generate_needle(needle_type)
            
            # Generate haystack content
            haystack_content = self._generate_haystack(target_length, needle_text, position_type)
            
            # Create question about the needle
            question = self._generate_question(needle_type, needle_value)
            
            # Determine difficulty based on context length and position
            difficulty = self._determine_difficulty(target_length, position_type)
            
            # Create evaluation metrics
            metrics = ["exact_match", "needle_retrieval", "position_accuracy"]
            if target_length > 32000:
                metrics.append("infinite_context")
            
            return BenchmarkTask(
                task_id=f"needle_haystack_{task_id}",
                task_type="needle_haystack",
                dataset_name="needle_in_haystack",
                context=haystack_content,
                prompt=question,
                expected_output=needle_value,
                evaluation_metrics=metrics,
                difficulty_level=difficulty,
                metadata={
                    "needle_type": needle_type,
                    "needle_value": needle_value,
                    "needle_text": needle_text,
                    "position_type": position_type,
                    "target_length": target_length,
                    "actual_length": len(haystack_content.split())
                }
            )
            
        except Exception as e:
            bt.logging.error(f"Error generating needle task: {e}")
            return None
    
    def _generate_needle(self, needle_type: str) -> tuple[str, str]:
        """Generate needle content based on type"""
        
        if needle_type == 'fact':
            facts = [
                "water boils at 100 degrees Celsius",
                "the Earth orbits the Sun once per year",
                "DNA contains genetic information",
                "photosynthesis produces oxygen",
                "gravity accelerates objects at 9.8 m/s²"
            ]
            needle_value = random.choice(facts)
            
        elif needle_type == 'number':
            needle_value = str(random.randint(10000, 99999))
            
        elif needle_type == 'name':
            names = [
                "Alexander Hamilton", "Marie Curie", "Leonardo da Vinci",
                "Cleopatra", "Albert Einstein", "Maya Angelou",
                "Nikola Tesla", "Frida Kahlo", "Charles Darwin"
            ]
            needle_value = random.choice(names)
            
        elif needle_type == 'date':
            year = random.randint(1800, 2023)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            needle_value = f"{month}/{day}/{year}"
            
        elif needle_type == 'location':
            locations = [
                "Mount Everest", "Sahara Desert", "Amazon Rainforest",
                "Great Barrier Reef", "Mariana Trench", "Antarctica",
                "Yellowstone National Park", "Machu Picchu"
            ]
            needle_value = random.choice(locations)
            
        elif needle_type == 'quote':
            quotes = [
                "The only way to do great work is to love what you do",
                "Innovation distinguishes between a leader and a follower",
                "The future belongs to those who believe in the beauty of their dreams",
                "Success is not final, failure is not fatal: it is the courage to continue that counts"
            ]
            needle_value = random.choice(quotes)
            
        else:
            needle_value = f"special_value_{random.randint(1000, 9999)}"
        
        # Create needle text using template
        template = random.choice(self.needle_templates.get(needle_type, ["{value}"]))
        needle_text = template.format(value=needle_value)
        
        return needle_value, needle_text
    
    def _generate_haystack(self, target_length: int, needle_text: str, position_type: str) -> str:
        """Generate haystack content with embedded needle"""
        
        # Generate base content
        topic = random.choice(self.haystack_topics)
        base_content = self._generate_topic_content(topic, target_length)
        
        # Split into words for insertion
        words = base_content.split()
        needle_words = needle_text.split()
        
        # Determine needle position
        total_words = len(words)
        
        if position_type == 'beginning':
            insert_pos = random.randint(10, min(100, total_words // 10))
        elif position_type == 'early':
            insert_pos = random.randint(total_words // 10, total_words // 4)
        elif position_type == 'middle':
            insert_pos = random.randint(total_words // 3, 2 * total_words // 3)
        elif position_type == 'late':
            insert_pos = random.randint(3 * total_words // 4, 9 * total_words // 10)
        else:  # end
            insert_pos = random.randint(max(total_words - 100, 9 * total_words // 10), total_words - 10)
        
        # Insert needle
        words[insert_pos:insert_pos] = needle_words
        
        # Ensure target length
        if len(words) < target_length:
            # Add more content
            additional_content = self._generate_topic_content(topic, target_length - len(words))
            words.extend(additional_content.split())
        elif len(words) > target_length:
            # Truncate while preserving needle
            words = words[:target_length]
        
        return ' '.join(words)
    
    def _generate_topic_content(self, topic: str, word_count: int) -> str:
        """Generate content about a specific topic"""
        
        # Base content templates for different topics
        content_templates = {
            'technology': [
                "Technology continues to evolve rapidly in the modern world.",
                "Artificial intelligence and machine learning are transforming industries.",
                "Computer systems process vast amounts of data every second.",
                "Software development requires careful planning and execution.",
                "Digital transformation affects every aspect of business operations."
            ],
            'science': [
                "Scientific research advances our understanding of the natural world.",
                "Experiments provide evidence for theoretical predictions.",
                "The scientific method ensures reliable and reproducible results.",
                "Researchers collaborate across disciplines to solve complex problems.",
                "Scientific discoveries often lead to practical applications."
            ],
            'history': [
                "Historical events shape the course of human civilization.",
                "Understanding the past helps us make sense of the present.",
                "Archaeological evidence reveals details about ancient cultures.",
                "Historical documents provide insights into past societies.",
                "The study of history teaches valuable lessons for the future."
            ]
        }
        
        # Get base sentences for the topic
        base_sentences = content_templates.get(topic, [
            "This is general content about various topics.",
            "Information can be found in many different sources.",
            "Knowledge comes from careful study and observation.",
            "Learning requires dedication and persistent effort.",
            "Understanding develops through experience and practice."
        ])
        
        # Generate content by repeating and varying base sentences
        content_words = []
        while len(content_words) < word_count:
            sentence = random.choice(base_sentences)
            # Add some variation
            if random.random() < 0.3:
                sentence = sentence.replace("the", "a").replace("The", "A")
            content_words.extend(sentence.split())
        
        return ' '.join(content_words[:word_count])
    
    def _generate_question(self, needle_type: str, needle_value: str) -> str:
        """Generate question about the needle"""
        
        question_templates = {
            'fact': [
                "What important fact was mentioned in the text?",
                "What key information should be remembered?",
                "What crucial fact was stated?"
            ],
            'number': [
                "What was the magic number mentioned?",
                "What secret code was given?",
                "What important number was stated?"
            ],
            'name': [
                "What person was mentioned as important to remember?",
                "Which individual was identified as key?",
                "What name should be remembered?"
            ],
            'date': [
                "What crucial date was mentioned?",
                "What important date should be remembered?",
                "What significant date was given?"
            ],
            'location': [
                "What important location was mentioned?",
                "What key place should be remembered?",
                "What crucial location was stated?"
            ],
            'quote': [
                "What important quote was mentioned?",
                "What key statement should be remembered?",
                "What significant quote was given?"
            ]
        }
        
        templates = question_templates.get(needle_type, ["What important information was mentioned?"])
        return random.choice(templates)
    
    def _determine_difficulty(self, context_length: int, position_type: str) -> str:
        """Determine task difficulty based on context length and needle position"""
        
        # Base difficulty from context length
        if context_length > 64000:
            base_difficulty = "extreme"
        elif context_length > 32000:
            base_difficulty = "hard"
        elif context_length > 16000:
            base_difficulty = "medium"
        else:
            base_difficulty = "easy"
        
        # Adjust for position difficulty
        position_difficulty = {
            'beginning': 0,
            'early': 1,
            'middle': 2,
            'late': 1,
            'end': 0
        }
        
        difficulty_levels = ["easy", "medium", "hard", "extreme"]
        base_index = difficulty_levels.index(base_difficulty)
        position_adjustment = position_difficulty.get(position_type, 0)
        
        final_index = min(len(difficulty_levels) - 1, base_index + position_adjustment)
        return difficulty_levels[final_index]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about needle-in-haystack configuration"""
        
        return {
            "loader_type": "needle_haystack",
            "is_synthetic": True,
            "context_length_range": (self.min_context_length, self.max_context_length),
            "needle_types": self.needle_types,
            "position_types": self.position_types,
            "haystack_topics": self.haystack_topics,
            "total_needle_templates": sum(len(templates) for templates in self.needle_templates.values())
        }