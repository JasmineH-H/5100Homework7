"""
Cryptographic Problem Solver
A comprehensive tool for solving various cryptographic puzzles and ciphers
with heuristics and efficiency optimizations.
"""

import re
import string
import itertools
from collections import Counter, defaultdict
import math
import random
from typing import List, Tuple, Dict, Optional, Set
import time

class CryptographicSolver:
    def __init__(self):
        # English letter frequency analysis (most common to least common)
        self.english_freq = {
            'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75,
            'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78,
            'U': 2.76, 'M': 2.41, 'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97,
            'P': 1.93, 'B': 1.29, 'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15,
            'Q': 0.10, 'Z': 0.07
        }
        
        # Common English words for validation
        self.common_words = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER',
            'WAS', 'ONE', 'OUR', 'HAD', 'HAS', 'HOW', 'MAN', 'NEW', 'NOW', 'OLD',
            'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE',
            'TOO', 'USE', 'HIS', 'WAY', 'MAY', 'DAY', 'GET', 'OWN', 'SAW', 'HIM'
        }
        
        # Common bigrams and trigrams for additional validation
        self.common_bigrams = {'TH', 'HE', 'IN', 'ER', 'AN', 'RE', 'ED', 'ND', 'ON', 'EN'}
        self.common_trigrams = {'THE', 'AND', 'ING', 'HER', 'HAT', 'HIS', 'THA', 'ERE', 'FOR', 'ENT'}

    def calculate_fitness_score(self, text: str) -> float:
        """
        Calculate fitness score based on English language characteristics.
        Higher score indicates more English-like text.
        Enhanced version with better discrimination.
        """
        if not text:
            return 0
        
        text = text.upper()
        score = 0
        
        # Letter frequency analysis (weighted heavily)
        letter_counts = Counter(c for c in text if c.isalpha())
        total_letters = sum(letter_counts.values())
        
        if total_letters == 0:
            return 0
        
        # Compare against expected English frequencies with better weighting
        freq_score = 0
        for letter in string.ascii_uppercase:
            expected_freq = self.english_freq.get(letter, 0.01)
            actual_count = letter_counts.get(letter, 0)
            actual_freq = (actual_count / total_letters) * 100
            # Use chi-squared-like scoring for better discrimination
            if expected_freq > 0:
                freq_score += (actual_freq - expected_freq) ** 2 / expected_freq
        
        # Convert to positive score (lower chi-squared = better fit)
        score += max(0, 100 - freq_score)
        
        # Bonus for common words (increased weight)
        words = re.findall(r'[A-Z]+', text)
        word_score = sum(15 for word in words if word in self.common_words)
        # Extra bonus for longer common words
        word_score += sum(5 * len(word) for word in words 
                         if word in self.common_words and len(word) > 3)
        score += word_score
        
        # Bonus for common bigrams and trigrams (increased weight)
        bigram_score = sum(8 for i in range(len(text)-1) 
                          if text[i:i+2] in self.common_bigrams)
        trigram_score = sum(12 for i in range(len(text)-2) 
                           if text[i:i+3] in self.common_trigrams)
        
        score += bigram_score + trigram_score
        
        # Penalty for unusual letter combinations
        penalty = 0
        for i in range(len(text)-1):
            if text[i:i+2] in ['QU', 'X']: # QU is good, but double letters might be suspicious
                continue
            # Penalize very uncommon bigrams
            bigram = text[i:i+2]
            if bigram.isalpha() and bigram not in self.common_bigrams:
                # Check if it's a really unusual combination
                if bigram in ['QX', 'XQ', 'JX', 'XJ', 'ZX', 'XZ', 'QZ', 'ZQ']:
                    penalty += 5
        
        score -= penalty
        
        return score / len(text)  # Normalize by text length

    def solve_caesar_cipher(self, ciphertext: str) -> List[Tuple[int, str, float]]:
        """
        Solve Caesar cipher by trying all possible shifts.
        Returns list of (shift, decrypted_text, fitness_score) sorted by fitness.
        """
        results = []
        
        for shift in range(26):
            decrypted = self.caesar_decrypt(ciphertext, shift)
            fitness = self.calculate_fitness_score(decrypted)
            results.append((shift, decrypted, fitness))
        
        # Sort by fitness score (descending)
        return sorted(results, key=lambda x: x[2], reverse=True)

    def caesar_decrypt(self, text: str, shift: int) -> str:
        """Decrypt text using Caesar cipher with given shift."""
        result = ""
        for char in text:
            if char.isalpha():
                ascii_offset = 65 if char.isupper() else 97
                shifted = (ord(char) - ascii_offset - shift) % 26
                result += chr(shifted + ascii_offset)
            else:
                result += char
        return result

    def solve_substitution_cipher(self, ciphertext: str, max_iterations: int = 20000) -> Tuple[str, Dict[str, str], float]:
        """
        Solve monoalphabetic substitution cipher using frequency analysis
        and hill climbing optimization with multiple restart heuristic.
        """
        best_solution = ("", {}, 0)
        
        # Try multiple random starting points (restart heuristic)
        # Increase restarts for better coverage of solution space
        for restart in range(10):
            current_key = self.generate_initial_substitution_key(ciphertext)
            current_text = self.apply_substitution(ciphertext, current_key)
            current_fitness = self.calculate_fitness_score(current_text)
            
            # Hill climbing with random mutations
            iterations_per_restart = max_iterations // 10
            for iteration in range(iterations_per_restart):
                # Generate neighbor by swapping two random letters
                new_key = current_key.copy()
                letters = list(string.ascii_uppercase)
                
                # Sometimes do single swaps, sometimes multiple swaps for larger jumps
                if random.random() < 0.8:  # 80% single swap
                    a, b = random.sample(letters, 2)
                    if a in new_key and b in new_key:
                        new_key[a], new_key[b] = new_key[b], new_key[a]
                else:  # 20% multiple swaps for larger exploration
                    swap_pairs = random.randint(2, 4)
                    for _ in range(swap_pairs):
                        a, b = random.sample(letters, 2)
                        if a in new_key and b in new_key:
                            new_key[a], new_key[b] = new_key[b], new_key[a]
                
                new_text = self.apply_substitution(ciphertext, new_key)
                new_fitness = self.calculate_fitness_score(new_text)
                
                # Accept if better (greedy) or with small probability if worse (simulated annealing)
                temperature = max(0.05, 1.0 - (iteration / iterations_per_restart))
                if new_fitness > current_fitness or (
                    random.random() < math.exp((new_fitness - current_fitness) / (temperature * 10))
                ):
                    current_key = new_key
                    current_text = new_text
                    current_fitness = new_fitness
            
            if current_fitness > best_solution[2]:
                best_solution = (current_text, current_key, current_fitness)
        
        return best_solution

    def generate_initial_substitution_key(self, ciphertext: str) -> Dict[str, str]:
        """
        Generate initial substitution key using frequency analysis heuristic.
        """
        # Count letter frequencies in ciphertext
        cipher_freq = Counter(c.upper() for c in ciphertext if c.isalpha())
        
        # Sort by frequency (most common first)
        sorted_cipher = [letter for letter, _ in cipher_freq.most_common()]
        sorted_english = sorted(self.english_freq.keys(), 
                              key=lambda x: self.english_freq[x], reverse=True)
        
        # Map most frequent cipher letters to most frequent English letters
        key = {}
        for i, cipher_letter in enumerate(sorted_cipher):
            if i < len(sorted_english):
                key[cipher_letter] = sorted_english[i]
        
        # Fill remaining letters randomly
        used_english = set(key.values())
        remaining_english = [c for c in string.ascii_uppercase if c not in used_english]
        remaining_cipher = [c for c in string.ascii_uppercase if c not in key]
        
        import random
        random.shuffle(remaining_english)
        for i, cipher_letter in enumerate(remaining_cipher):
            if i < len(remaining_english):
                key[cipher_letter] = remaining_english[i]
        
        return key

    def apply_substitution(self, text: str, key: Dict[str, str]) -> str:
        """Apply substitution cipher with given key."""
        result = ""
        for char in text:
            if char.upper() in key:
                mapped = key[char.upper()]
                result += mapped if char.isupper() else mapped.lower()
            else:
                result += char
        return result

    def solve_vigenere_cipher(self, ciphertext: str, max_key_length: int = 10) -> List[Tuple[str, str, float]]:
        """
        Solve Vigenère cipher using Kasiski examination and frequency analysis.
        """
        results = []
        
        # Try different key lengths (length-based heuristic)
        for key_length in range(1, min(max_key_length + 1, len(ciphertext) // 3)):
            key = self.find_vigenere_key(ciphertext, key_length)
            if key:
                decrypted = self.vigenere_decrypt(ciphertext, key)
                fitness = self.calculate_fitness_score(decrypted)
                results.append((key, decrypted, fitness))
        
        return sorted(results, key=lambda x: x[2], reverse=True)

    def find_vigenere_key(self, ciphertext: str, key_length: int) -> Optional[str]:
        """Find Vigenère key of given length using frequency analysis."""
        cipher_only = ''.join(c.upper() for c in ciphertext if c.isalpha())
        
        if len(cipher_only) < key_length:
            return None
        
        key = ""
        
        # For each position in the key
        for i in range(key_length):
            # Extract every key_length-th character
            column = [cipher_only[j] for j in range(i, len(cipher_only), key_length)]
            
            if not column:
                continue
            
            # Find the shift that gives the best frequency match
            best_shift = 0
            best_score = -float('inf')
            
            for shift in range(26):
                shifted_column = [chr((ord(c) - ord('A') - shift) % 26 + ord('A')) for c in column]
                score = self.calculate_fitness_score(''.join(shifted_column))
                
                if score > best_score:
                    best_score = score
                    best_shift = shift
            
            key += chr(best_shift + ord('A'))
        
        return key

    def vigenere_decrypt(self, text: str, key: str) -> str:
        """Decrypt text using Vigenère cipher with given key."""
        result = ""
        key_index = 0
        
        for char in text:
            if char.isalpha():
                shift = ord(key[key_index % len(key)]) - ord('A')
                if char.isupper():
                    result += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
                else:
                    result += chr((ord(char) - ord('a') - shift) % 26 + ord('a'))
                key_index += 1
            else:
                result += char
        
        return result

    def detect_cipher_type(self, ciphertext: str) -> str:
        """
        Heuristic to detect likely cipher type based on text characteristics.
        """
        cipher_only = ''.join(c for c in ciphertext if c.isalpha())
        
        if not cipher_only:
            return "unknown"
        
        # Calculate index of coincidence
        ic = self.calculate_index_of_coincidence(cipher_only)
        
        # Check for patterns that might indicate cipher type
        if ic > 0.06:  # Close to English (0.067)
            return "caesar"
        elif ic > 0.045:  # Moderate IC suggests substitution
            return "substitution"
        else:  # Low IC suggests polyalphabetic (Vigenère)
            return "vigenere"

    def calculate_index_of_coincidence(self, text: str) -> float:
        """Calculate index of coincidence for the text."""
        text = text.upper()
        n = len(text)
        
        if n <= 1:
            return 0
        
        freq = Counter(text)
        ic = sum(f * (f - 1) for f in freq.values()) / (n * (n - 1))
        return ic

    def solve_cipher(self, ciphertext: str, cipher_type: str = None) -> Dict:
        """
        Main solving function that automatically detects and solves various ciphers.
        """
        start_time = time.time()
        
        if not cipher_type:
            cipher_type = self.detect_cipher_type(ciphertext)
        
        print(f"Detected/Using cipher type: {cipher_type}")
        
        results = {
            'cipher_type': cipher_type,
            'solutions': [],
            'processing_time': 0
        }
        
        try:
            if cipher_type == "caesar":
                solutions = self.solve_caesar_cipher(ciphertext)
                results['solutions'] = [(f"Shift {s}", text, score) for s, text, score in solutions[:3]]
            
            elif cipher_type == "substitution":
                text, key, score = self.solve_substitution_cipher(ciphertext)
                key_str = ''.join(f"{k}->{v} " for k, v in sorted(key.items()))
                results['solutions'] = [(f"Key: {key_str}", text, score)]
            
            elif cipher_type == "vigenere":
                solutions = self.solve_vigenere_cipher(ciphertext)
                results['solutions'] = [(f"Key: {key}", text, score) for key, text, score in solutions[:3]]
            
            else:
                # Try all methods if type is unknown
                caesar_results = self.solve_caesar_cipher(ciphertext)
                if caesar_results:
                    shift, text, score = caesar_results[0]
                    results['solutions'].append((f"Caesar Shift {shift}", text, score))
                
                # Note: For efficiency, we might skip substitution for very long texts
                if len(ciphertext) < 500:
                    sub_text, sub_key, sub_score = self.solve_substitution_cipher(ciphertext, 1000)
                    results['solutions'].append(("Substitution", sub_text, sub_score))
                
                vigenere_results = self.solve_vigenere_cipher(ciphertext, 6)
                if vigenere_results:
                    key, text, score = vigenere_results[0]
                    results['solutions'].append((f"Vigenère Key: {key}", text, score))
        
        except Exception as e:
            results['error'] = str(e)
        
        results['processing_time'] = time.time() - start_time
        return results

# Example usage and testing
if __name__ == "__main__":
    solver = CryptographicSolver()
    
    # Test cases
    test_cases = [
        ("WKLV LV D WHVW PHVVDJH", "caesar"),  # Caesar cipher with shift 3: "THIS IS A TEST MESSAGE"
        ("FRPHW DIWHU PH", "caesar"),  # "COMET AFTER ME" with shift 3
        
        # Monoalphabetic substitution cipher example - SHORT (challenging)
        # Original: "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
        # Key: A->Q, B->W, C->E, D->R, E->T, F->Y, G->U, H->I, I->O, J->P, K->A, L->S, M->D, N->F, O->G, P->H, Q->J, R->K, S->L, T->Z, U->X, V->C, W->V, X->B, Y->N, Z->M
        ("ZIT JXOEA WKGVF YGB PXDHL GCTK ZIT SQMN RGU", "substitution"),
        
        ("WKH TXLFN EURZQ IRA MXPSV RYHU WKH ODCB GRJ", None),  # Auto-detect (Caesar shift 3)
        
        # Vigenère cipher example
        # Short example (challenging due to length): "HELLO WORLD" with key "KEY" 
        ("RIJVS UYVJN", "vigenere"),
    ]
    
    for ciphertext, cipher_type in test_cases:
        print(f"\n{'='*60}")
        print(f"Solving: {ciphertext}")
        print(f"{'='*60}")
        
        results = solver.solve_cipher(ciphertext, cipher_type)
        
        print(f"Cipher type: {results['cipher_type']}")
        print(f"Processing time: {results['processing_time']:.3f} seconds")
        
        if 'error' in results:
            print(f"Error: {results['error']}")
        else:
            print("\nTop solutions:")
            for i, (method, solution, score) in enumerate(results['solutions'], 1):
                print(f"{i}. {method}")
                print(f"   Text: {solution}")
                print(f"   Fitness Score: {score:.2f}")
                print()