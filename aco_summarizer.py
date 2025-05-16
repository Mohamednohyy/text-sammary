"""
Simplified Ant Colony Optimization (ACO) algorithm for extractive text summarization.

This module implements a basic version of the ACO algorithm to select important sentences
from a document to create a concise summary. Designed for educational purposes.
"""

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import re
import random


class ACOSummarizer:
    def __init__(self, num_ants=10, alpha=1.0, beta=2.0, rho=0.1, q0=0.9, 
                 max_iterations=30, compression_ratio=0.3):
        """
        Initialize the ACO Summarizer with simplified parameters.
        
        Parameters:
        -----------
        num_ants : int
            Number of ants in the colony (more ants = more exploration)
        alpha : float
            Importance of pheromone (higher = follow other ants more)
        beta : float
            Importance of heuristic information (higher = greedier selection)
        rho : float
            Pheromone evaporation rate (how quickly trails fade)
        q0 : float
            Probability of exploitation vs exploration (higher = more greedy)
        max_iterations : int
            Maximum number of iterations (more = better but slower)
        compression_ratio : float
            Target summary length as a fraction of original document length
        """
        # Store parameters
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.max_iterations = max_iterations
        self.compression_ratio = compression_ratio
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading required NLTK resources...")
            nltk.download('punkt')
            nltk.download('stopwords')
            
        # Load stopwords for cleaning
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """
        Preprocess the text by tokenizing into sentences and cleaning.
        
        Parameters:
        -----------
        text : str
            The input document text
            
        Returns:
        --------
        list
            List of preprocessed sentences
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        # Clean sentences
        cleaned_sentences = []
        for sentence in sentences:
            # Remove special characters and numbers
            sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
            # Convert to lowercase
            sentence = sentence.lower()
            # Remove extra whitespace
            sentence = re.sub(r'\s+', ' ', sentence).strip()
            
            if sentence:  # Only add non-empty sentences
                cleaned_sentences.append(sentence)
                
        return sentences, cleaned_sentences
    
    def create_sentence_vectors(self, cleaned_sentences):
        """
        Create simple bag-of-words vectors for sentences.
        
        Parameters:
        -----------
        cleaned_sentences : list
            List of cleaned sentences
            
        Returns:
        --------
        list
            List of sentence feature vectors
        """
        # Create a vocabulary of all unique words
        all_words = set()
        for sentence in cleaned_sentences:
            words = sentence.split()
            # Remove stop words
            words = [word for word in words if word not in self.stop_words]
            all_words.update(words)
        
        vocabulary = list(all_words)
        
        # Create sentence vectors (1 if word is present, 0 if not)
        sentence_vectors = []
        for sentence in cleaned_sentences:
            words = sentence.split()
            words = [word for word in words if word not in self.stop_words]
            
            # Create a vector for this sentence
            vector = [1 if word in words else 0 for word in vocabulary]
            sentence_vectors.append(vector)
            
        return sentence_vectors
    

    
    def calculate_sentence_scores(self, sentences, cleaned_sentences):
        """
        Calculate importance scores for each sentence based on multiple features.
        
        Parameters:
        -----------
        sentences : list
            Original sentences
        cleaned_sentences : list
            Cleaned sentences
            
        Returns:
        --------
        numpy.ndarray
            Array of sentence scores
        """
        n = len(sentences)
        scores = np.zeros(n)
        
        # Create sentence vectors
        sentence_vectors = self.create_sentence_vectors(cleaned_sentences)
        
        # If we have empty vectors, return zeros
        if not sentence_vectors or len(sentence_vectors[0]) == 0:
            return scores
            
        # Convert to numpy array
        sentence_vectors = np.array(sentence_vectors)
        
        # Calculate similarity between sentences
        similarity_matrix = cosine_similarity(sentence_vectors)
        
        # Feature 1: Position score (first and last sentences are important)
        position_scores = np.zeros(n)
        for i in range(n):
            # Sentences at the beginning and end get higher scores
            position_scores[i] = 1.0 - abs(i - n/2) / (n/2)
        
        # Feature 2: Length score (not too short, not too long)
        length_scores = np.zeros(n)
        for i in range(n):
            words = cleaned_sentences[i].split()
            length = len(words)
            # Penalize very short or very long sentences
            if length < 3:
                length_scores[i] = 0.3
            elif length > 30:
                length_scores[i] = 0.5
            else:
                length_scores[i] = 1.0
        
        # Feature 3: Centrality score (how similar a sentence is to others)
        centrality_scores = np.sum(similarity_matrix, axis=1) / n
        
        # Combine all features with weights
        for i in range(n):
            scores[i] = (
                0.3 * position_scores[i] +  # Position weight
                0.2 * length_scores[i] +    # Length weight
                0.5 * centrality_scores[i]  # Centrality weight
            )
        
        # Normalize scores to [0, 1]
        if np.max(scores) > 0:
            scores = scores / np.max(scores)
        
        return scores
    
    def initialize_pheromones(self, n):
        """
        Initialize pheromone matrix.
        
        Parameters:
        -----------
        n : int
            Number of sentences
            
        Returns:
        --------
        numpy.ndarray
            Pheromone matrix
        """
        # Initialize with a small constant value
        return np.ones((n, n)) * 0.1
    
    def ant_solution(self, n, heuristic_scores, pheromones, target_length):
        """
        Generate a solution by a single ant.
        
        This simulates an ant traversing the document and selecting sentences.
        
        Parameters:
        -----------
        n : int
            Number of sentences
        heuristic_scores : numpy.ndarray
            Heuristic scores for sentences
        pheromones : numpy.ndarray
            Pheromone matrix
        target_length : int
            Target number of sentences in summary
            
        Returns:
        --------
        list
            Selected sentence indices
        float
            Solution quality
        """
        # Start with a random sentence
        current = random.randint(0, n-1)
        selected = [current]
        
        # Select remaining sentences until we reach target length
        while len(selected) < target_length:
            # Calculate probability of selecting each sentence
            probabilities = np.zeros(n)
            
            for j in range(n):
                if j not in selected:  # Only consider unselected sentences
                    # ACO formula: τ^α × η^β
                    # τ (tau) = pheromone level
                    # η (eta) = heuristic value
                    tau = pheromones[current, j]
                    eta = heuristic_scores[j]
                    
                    probabilities[j] = (tau ** self.alpha) * (eta ** self.beta)
            
            # Normalize probabilities
            sum_prob = np.sum(probabilities)
            if sum_prob > 0:
                probabilities = probabilities / sum_prob
            
            # Choose next sentence (exploitation or exploration)
            if random.random() < self.q0:
                # Exploitation: choose the best option
                next_sentence = np.argmax(probabilities)
            else:
                # Exploration: choose probabilistically
                next_sentence = np.random.choice(n, p=probabilities)
            
            selected.append(next_sentence)
            current = next_sentence
        
        # Calculate solution quality (sum of scores)
        quality = np.sum(heuristic_scores[selected])
        
        return selected, quality
    
    def update_pheromones(self, pheromones, all_solutions, best_solution, best_quality):
        """
        Update pheromone levels based on the best solution.
        
        Parameters:
        -----------
        pheromones : numpy.ndarray
            Pheromone matrix
        all_solutions : list
            List of all ant solutions
        best_solution : list
            Best solution found
        best_quality : float
            Quality of the best solution
            
        Returns:
        --------
        numpy.ndarray
            Updated pheromone matrix
        """
        # Step 1: Evaporation - reduce all pheromones
        pheromones = (1 - self.rho) * pheromones
        
        # Step 2: Deposit new pheromones for the best solution
        for i in range(len(best_solution) - 1):
            current = best_solution[i]
            next_sentence = best_solution[i + 1]
            # Add pheromone proportional to solution quality
            pheromones[current, next_sentence] += self.rho * best_quality
        
        return pheromones
    
    def summarize(self, text):
        """
        Generate a summary using ACO.
        
        Parameters:
        -----------
        text : str
            The input document text
            
        Returns:
        --------
        str
            The generated summary
        list
            Indices of selected sentences
        """
        # Step 1: Preprocess text
        original_sentences, cleaned_sentences = self.preprocess_text(text)
        n = len(original_sentences)
        
        if n == 0:
            return "", []
        
        # Step 2: Calculate target summary length (30% of original by default)
        target_length = max(1, int(n * self.compression_ratio))
        
        # Step 3: Calculate sentence importance scores
        heuristic_scores = self.calculate_sentence_scores(original_sentences, cleaned_sentences)
        
        # Step 4: Initialize pheromone matrix
        pheromones = np.ones((n, n)) * 0.1
        
        # Step 5: Initialize best solution tracking
        best_solution = None
        best_quality = 0
        
        # Step 6: Main ACO loop - multiple iterations of ant colony
        for iteration in range(self.max_iterations):
            all_solutions = []
            
            # Each ant finds a solution
            for ant in range(self.num_ants):
                # Get this ant's solution
                solution, quality = self.ant_solution(n, heuristic_scores, pheromones, target_length)
                all_solutions.append((solution, quality))
                
                # Update best solution if better
                if quality > best_quality:
                    best_solution = solution
                    best_quality = quality
            
            # Update pheromones based on results
            pheromones = self.update_pheromones(pheromones, all_solutions, best_solution, best_quality)
        
        # Step 7: Generate final summary from best solution
        if best_solution:
            # Sort selected sentences by position in original text
            best_solution.sort()
            
            # Extract selected sentences and join them
            summary_sentences = [original_sentences[i] for i in best_solution]
            summary = ' '.join(summary_sentences)
            
            return summary, best_solution
        else:
            # Fallback: select top sentences by heuristic score
            top_indices = np.argsort(heuristic_scores)[-target_length:]
            top_indices.sort()
            summary_sentences = [original_sentences[i] for i in top_indices]
            summary = ' '.join(summary_sentences)
            
            return summary, list(top_indices)
