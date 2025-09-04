"""
Utility functions for handling bitstrings (necessary for error corecting and serializing signature data)

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""

def pad_bitstring(b, n):
    """
    Pads the bitstring b with zeros at the end to make its length a multiple of n.

    Args:
        b (str): The bitstring to pad, containing only '0's and '1's.
        n (int): The desired multiple for the length of the bitstring.
    
    Returns:
        b (str): The padded bitstring
    """
    #if b has number of bits that is not multiple of n, pad end with 0
    if len(b) % n != 0 :
        num_pad_bits = n - (len(b) % n)
        for i in range(num_pad_bits):
            b += "0"
    return b


def bitstring_to_ascii(b):
    """
    Converts a bitstring to an ASCII string.

    Args:
        b (str): The bitstring to convert, containing only '0's and '1's.

    Returns:
        out (str): The resulting ASCII string.
    """
    b = pad_bitstring(b, 8)

    i = 0
    out = ""
    while i + 8 < len(b) + 8:
        out += chr(int(b[i:i+8], 2))
        i += 8
    return out


def ascii_to_bitstring(ascii):
    """
    Converts an ASCII string to a bitstring.

    Args:
        ascii (str): The ASCII string to convert.
        
    Returns:
        out (str): The resulting bitstring.
    """
    out = ""
    for i in ascii:
        bits = bin(ord(i))[2:]
        if len(bits) < 8:
            for j in range(8 - len(bits)):
                bits = "0" + bits
        out += bits
    return out


def bitstring_to_bytes(bitstring):
    """
    Converts a bitstring to bytes object.

    Args:
        bitstring (str): The bitstring to convert, containing only '0's and '1's'.
    
    Returns:
        The resulting bytes object.
    """
    bitstring = pad_bitstring(bitstring, 8)
    return int(bitstring, 2).to_bytes((len(bitstring) + 7) // 8, 'big')


def bytes_to_bitstring(bytes_str):
    """
    Converts a bytes object to a bitstring.

    Args:
        bytes_str (bytes): The bytes object to convert.

    Returns:
        out (str): The resulting bitstring.
    """
    out = ""
    for c in bytes_str:
        bits = bin(c)[2:]
        if len(bits) < 8:
            for j in range(8 - len(bits)):
                bits = "0" + bits
        out += bits
    return out
