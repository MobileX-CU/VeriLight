"""
Random Projection-based LSH. Implementation found online at https://github.com/gamboviol/lsh.

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025
"""
import numpy as np
import random

class CosineHash:
    """
    Hashing function based on random projection, preserving cosine similarity.
    """
    def __init__(self,r):
        self.r = r

    def hash(self, vec):
        """
        Hash a vector using the cosine hash function.

        Args:
            vec (list or numpy array): The vector to be hashed.
        
        Returns:
            int: The hash value, either 0 or 1.
        """
        return self.sgn(np.dot(vec,self.r))

    def sgn(self, x):
        """
        Sign function.
        """
        return int(x>0)


class CosineHashFamily:
    """
    Family of CosineHash functions.
    """
    def __init__(self, d):
        self.d = d

    def create_hash_func(self):
        """
        Create a new CosineHash function.
        """
        return CosineHash(self.rand_vec())

    def rand_vec(self):
        """
        Create a random vector with each component drawn from a standard normal distribution.
        """
        return [random.gauss(0,1) for i in range(self.d)]

    def combine(self, hashes):
        """
        Combine multiple hash values into a single integer hash value.
        """
        return sum(2**i if h > 0 else 0 for i,h in enumerate(hashes))

    def concat(self, hashes):
        """
        Concatenate multiple hash values into a single bit string.
        """
        bitstr = ""
        for i in hashes:
            bitstr += str(i)
        return bitstr

def hamming(s1, s2):
    """
    Calculate the Hamming distance between two bit strings
    Notes: https://stackoverflow.com/questions/31007054/hamming-distance-between-two-binary-strings-not-working

    Args:
        s1 (str): First bit string
        s2 (str): Second bit string
    
    Returns:
        int: Hamming distance between s1 and s2
    """
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def hash_point(fam, hash_funcs, p):
    """
    Hash a vec/point using a family of hash functions.

    Args:
        fam (CosineHashFamily): Family of hash functions to use
        hash_funcs (list of CosineHash): List of hash functions to use
        p (list or numpy array): The vector/point to be hashed

    Returns:
        str: The concatenated bit string of hash values
    """
    return fam.concat([h.hash(p) for h in hash_funcs])


def get_lsh_hash_dist(v1, v2, k):
    """
    Assume v1 and v2 have same dimensionality
    Returns Hamming distance between Cosine famile LSH hashes of v1 and v2

    Args:
        v1 (numpy array): First vector
        v2 (numpy array): Second vector
        k (int): Number of hash functions to use
    
    Returns:
        int: Hamming distance between the LSH hashes of v1 and v2
    """
    d = v1.shape[0]
    fam = CosineHashFamily(d)
    hash_funcs = [fam.create_hash_func() for h in range(k)]
    h1 = hash_point(fam, hash_funcs, v1)
    h2 = hash_point(fam, hash_funcs, v2)
    d = hamming(h1, h2)
    return d







