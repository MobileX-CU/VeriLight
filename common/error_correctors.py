"""
Various error correction code: Reed Solomon, Viterbi, Concatenated Reed Solomon + Viterbi (used in paper)

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""
import unireedsolomon as rs
import random
import numpy as np

from common.viterbi import bit_encoder_K2, bit_encoder_K5, stream_encoder, Decoder
from common.bitstring_utils import bitstring_to_ascii, ascii_to_bitstring

#######################
# REED 
########################
class ReedSolomon(object):
    """
    Reed Solomon error correction code
    Wrapper around unireedsolomon library: https://pypi.org/project/unireedsolomon/
    """
    def __init__(self, n, k):
        """
        Args:
            n (int): codeword size (a codeword consists of n bytes, k of which are data and n - k of which are parity)
            k (int): number of data bytes
        """
        self.n = n
        self.k = k
        self.coder = rs.RSCoder(n, k)
    
    def strength(self):
        """
        Returns:
            int: number of symbol errors that can be corrected by this Reed Solomon code
            Based on https://www.cs.cmu.edu/~guyb/realworld/reedsolomon/reed_solomon_codes.html
        """
        return (self.n - self.k)/2 #number of error symbols that can be corrected

    def bitstring_to_codewords(self, bitstring):
        """
        Converts a bitstring into a list of codewords. Each codeword is represented as an ASCII string of length n.
        Args:
            bitstring (str): str containing only 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not
                             a char "1" or "0"
        Returns:
            list: list of codewords, where each codeword is represented as an ASCII string of length n
        """
        chunk_size = self.n * 8
        codewords = []
        for i in range(0, len(bitstring), chunk_size):
            chunk = bitstring[i:i+chunk_size]
            codeword = bitstring_to_ascii(chunk)
            codewords.append(codeword)
        return codewords

    def codewords_to_bitstring(self, codewords):
        """
        Converts a list of codewords into a bitstring.

        Args:
            codewords (list): list of codewords, where each codeword is represented as an ASCII string of length n
        Returns:
            str: str containing only 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not
                 a char "1" or "0"
        """
        bitstring = ""
        for c in codewords:
            bitstring += ascii_to_bitstring(c)
        return bitstring

    def encode(self, message_bitstring):
        """
        Code a message bitstring via Reed Solomon encoding.
        Args:
            message_bitstring (str): str containing only 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not
                                      a char "1" or "0"

        Returns:
            list: list of codewords, where each codeword is represented as an ASCII string of length n
        """
        #convert message bitstring to ascii for compatibility with the ReedSolomon coder 
        # which expects input in the form of 8-bit symbols 
        message_symbols = bitstring_to_ascii(message_bitstring)

        #split symbols into chunks of size n and obtain codeword for each 
        #if the last chunk has less than n bytes, pad it with 0x00 ASCII character until it is of size n
        codewords = []
        for i in range(0, len(message_symbols), self.k):
            if i + self.k > len(message_symbols):
                num_padding_symbols = self.k - (len(message_symbols) - i)
                chunk = message_symbols[i:]
                for n in range(num_padding_symbols):
                    chunk += chr(0)
            else:
                chunk = message_symbols[i:i+self.k]
            codeword = self.coder.encode(chunk)
            codewords.append(codeword)
        return codewords

    def encode_payload(self, message_bitsrting):
        """
        Encode a message bitstring via Reed Solomon encoding and return the encoded bitstring.
        
        Args:
            message_bitstring (str): str containing only 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not
                                      a char "1" or "0"

        Returns:
            str: the encoded bitstring, containing only 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not
                 a char "1" or "0"
        """
        codewords = self.encode(message_bitsrting)
        enc_bitstring = self.codewords_to_bitstring(codewords)
        return enc_bitstring

    def decode(self, codewords):
        """
        Decode a list of Reed Solomon codewords into a bitstring.

        Args:
            codewords (list): list of codewords, where each codeword is represented as an ASCII string of length n

        Returns:
            str: str containing only 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not
                 a char "1" or "0"
        """
        recovered_bitstring = ""
        correctable = True
        for i, c in enumerate(codewords):
            try:
                recovered_chunk_symbols = self.coder.decode(c, nostrip = True)[0]
            except Exception as e:
                # print(f"Reed Solomon can't corect codeword {i}. {e}. Recovered message will just be uncorrected data portion of codeword.")
                recovered_chunk_symbols = c[:self.k]
                correctable = False
            recovered_chunk_bitstring = ascii_to_bitstring(recovered_chunk_symbols)
            recovered_bitstring += recovered_chunk_bitstring
        return recovered_bitstring, correctable
    
    def decode_payload(self, encoded_bitstring):
        """
        Decode an encoded bitstring via Reed Solomon decoding and return the decoded bitstring.

        Args:
            encoded_bitstring (str): str containing only 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not
                                      a char "1" or "0"
        
        Returns:
            str: the decoded bitstring, containing only 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not
                 a char "1" or "0"
        """
        codewords = self.bitstring_to_codewords(encoded_bitstring)
        decoded_bitstring, correctable = self.decode(codewords)
        return decoded_bitstring, correctable

    def noise_codewords(self, codewords, flip_probability):
        """
        Add noise to a list of codewords by flipping bits with probability flip_probability, for testing purposes.

        Args:
            codewords (list): list of codewords, where each codeword is represented as an ASCII string of length n
            flip_probability (float): probability of flipping each bit in the codewords
        
        Returns:
            noised_codewords (list): list of noised codewords, where each codeword is represented as an ASCII string of length n
            num_bit_flips (int): total number of bit flips that were made across all codewords
        """
        #add noise to the codewords by flipping bits
        noised_codewords = []
        num_bit_flips = 0
        c_bitstrings = self.codewords_to_bitstrings(codewords)
        for c in c_bitstrings:
            noised_c_bitstring = ""
            for i in c:
                if random.randint(0, 100) < flip_probability*100:
                    num_bit_flips += 1
                    if i == "1":
                        noised_c_bitstring += "0"
                    else:
                        noised_c_bitstring += "1"
                else:
                    noised_c_bitstring += i
            noised_c = bitstring_to_ascii(noised_c_bitstring)
            noised_codewords.append(noised_c)
        return noised_codewords, num_bit_flips

    def add_payload_noise(self, encoded_payload, flip_probability):
        """
        Add noise to an encoded bitstring by flipping bits with probability flip_probability, for testing purposes.

        Args:
            encoded_payload (str): str containing only 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not
                                   a char "1" or "0"
            flip_probability (float): probability of flipping each bit in the encoded_payload
        
        Returns:
            noised_payload (str): str containing only 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not
        """
        num_bit_flips = 0
        noised_payload = ""
        for i in encoded_payload:
            if random.randint(0, 100) < flip_probability*100:
                num_bit_flips += 1
                if i == "1":
                    noised_payload += "0"
                else:
                    noised_payload += "1"
            else:
                noised_payload += i
        return noised_payload, num_bit_flips

    def check(self, test_message_bitstring):
        """
        Check the Reed Solomon encoding and decoding process by comparing the original
        message bitstring with the recovered bitstring after encoding and decoding.
        """
        codewords = self.encode(test_message_bitstring)
        recovered_bitstring = self.decode(codewords)
        err = sum(c1!=c2 for c1,c2 in zip(test_message_bitstring, recovered_bitstring))
        print("----- REED SOLOMON CHECK -----")
        print(f"Input bitstring:     {test_message_bitstring}")
        print(f"Recovered bitstring: {recovered_bitstring}")
        print(f"Total errors: {err} bits") 


#######################
# VITERBI
########################
class SoftViterbi(object):
    """
    Viterbi error correction code with soft decoding
    """
    def __init__(self, k):
        """
        Args:
            k (int): Viterbi encoding k. Must be either 2 or 5 right now
        """
        self.k = k
        if k == 2:
            self.bit_encoder = bit_encoder_K2
        elif k == 5:
            self.bit_encoder = bit_encoder_K5

    def encode_payload(self, input_bitstring):
        """
        Encode a bitstring via Viterbi encoding and return the encoded bitstring.

        Args:
            input_bitstring (str): str containing only 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not
                                        a char "1" or "0"
        
        Returns:
            str: the encoded bitstring, containing only 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not
                    a char "1" or "0"
        """
        input_stream = [int(i) for i in input_bitstring]
        list_output = stream_encoder(self.k, input_stream)
        encoded_bitstring = ""
        for el in list_output:
            encoded_bitstring += str(el[0])
            encoded_bitstring += str(el[1])
        return encoded_bitstring

    def decode_payload(self, input_probs):
        """
        Decode a Viterbi-encoded bitstring using soft decision decoding.

        Args:
            input_probs (list of floats): each float representing the probability of a bit being a 1
        
        Returns:
            dec_bitstring (str): the decoded bitstring, containing only 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not
                                  a char "1" or "0"
            correctable (bool): whether the Viterbi error correction was successful. 
        """
        input_stream = []
        correctable = True
        if len(input_probs) % 2 != 0 or len(input_probs) < self.k:
            correctable = False
        else:
            for i in range(0, len(input_probs), 2):
                input_stream.append([input_probs[i], input_probs[i + 1]])
            try:
                dec_list = Decoder(self.k, input_stream, False)
            except:
                correctable = False
        if not correctable:
            print(f"Viterbi decoder failed. Returning {(len(input_stream) - self.k)} 0s")
            dec_bitstring = "0" * (len(input_probs) - self.k)
        else:
            dec_bitstring = ""
            for b in dec_list:
                dec_bitstring += str(b)
        return dec_bitstring, correctable


############################
# CONCATENATED RS + VITERBI
############################
class ConcatenatedViterbiRS(object):
    """
    Concatenated error correction: first Viterbi, then RS on Viterbi-encoded
    """
    def __init__(self, v_k, n, rs_k):
        """
        Initialize a concatenated error correction code with an inner Viterbi 
        encoder and an outer Reed Solomon (RS) encoder

        Args:
            v_k (int): Viterbi encoding k
            n (int): Reed Solomon codeword size (a codeword consists of n bytes, 
                     k of which are data and n - k of which are parity)
            rs_k (int): number of data bytes input to Reed Solomon encoder
        """
        self.viterbi_coder = SoftViterbi(v_k)
        self.rs_coder = ReedSolomon(n, rs_k)
    
    def encode_payload(self, input_bitstring):
        """
        First do Viterbi encoding on input bitstring, then run Reed-Solomon 
        encoding on the Viterbi-encoded bitstring

        Args:
            input_bitstring (str): the input bitstring to be encoded. 
        
        Returns:
            final_encoded (str): the final encoded bitstring (after both Viterbi and RS encoding)
            rs_encoded (str): the Reed Solomon coded bitstring (i.e., pre-Viterbi encoding, post-RS encoding)

        Note: Here, a bitstring is a string of 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not a char
              e.g., "100010010010"
        """
        rs_encoded = self.rs_coder.encode_payload(input_bitstring)
        final_encoded = self.viterbi_coder.encode_payload(rs_encoded)
        return final_encoded, rs_encoded

    def decode_payload(self, input_probs):
        """
        First do soft Viterbi decoding on input probabilities to recover a Reed-Solomon codeword.
        Then run RS decoding on codeword to obtain final data

        Args:
            input_probs (list of floats): each float representing the probability of a bit being a 1
        
        Returns:
            decoded_bitstring (str): the decoded bitstring (after both Viterbi and RS decoding)
            pred_rs_encoded (str): the Reed Solomon coded bitstring (i.e., post-Viterbi decoding, pre-RS decoding)
            correctable (bool): whether the error correction was successful, considering any errors encountered Viterbi and/or RS decoding
        
        Note: Here, a bitstring is a string of 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not a char
              e.g., "100010010010"
        """
        pred_rs_encoded, vit_corectable = self.viterbi_coder.decode_payload(input_probs)
        decoded_bitstring, rs_correctable = self.rs_coder.decode_payload(pred_rs_encoded)
        correctable = vit_corectable and rs_correctable
        return decoded_bitstring, pred_rs_encoded, correctable

#######################
# TESTING/SIMULATION
########################
class ReedNoisyChannelSimulator(object):
    def __init__(self, error_corrector):
        """
        Accept either ReedMuller or ReedSolom class above
        """
        self.error_corrector = error_corrector
        if type(self.error_corrector) == ReedSolomon:
            self.error_corrector_name = "REED SOLOMON"
        else:
            self.error_corrector_name = "REED MULLER"
    
    def simulate(self, message_bitstring, error_probability):
        """
        Simulate sending a message bitstring through a noisy channel with the given error probability,
        using the specified error correction code to encode and decode the message.

        Args:
            message_bitstring (str): str containing only 0s and 1s, where each 0 and 1 is treated as an actual binary 1/0 value, not
                                        a char "1" or "0"
            error_probability (float): probability of flipping each bit in the encoded message during transmission
        
        Returns:
            None: prints the input bitstring, recovered bitstring, number of bits flipped during noise addition, and total number of errors
        """
        encoded_payload = self.error_corrector.encode_payload(message_bitstring)
        noised_payload, num_bit_flips = self.error_corrector.add_payload_noise(encoded_payload, error_probability)
        decoded_payload = self.error_corrector.decode_payload(noised_payload)
        err = sum(c1!=c2 for c1,c2 in zip(message_bitstring, decoded_payload))
        print(f"-----{self.error_corrector_name} NOISY CHANNEL SIMULATOR -----")
        print(f"Input bitstring:     {message_bitstring}")
        print(f"Recovered bitstring: {decoded_payload}")
        print(f"{num_bit_flips} bits flipped during noise addition.")
        print(f"Total errors: {err} bits")


#######################
# UTILS
########################
def get_rs_params(data_capacity, raw_signature_size, viterbi_k = None):
    """
    Get maximal Reed Solomon parameters n and k given data capacity of a window and raw signature size.
    If using concatenated error correction with Viterbi, account for Viterbi overhead.

    Args:
        data_capacity (int): data capacity of a single window, in bits
        signature_size (int): size of raw payload (i.e., just window and hash bits), in bits
        viterbi_k (int, optional): k associated with Viterbi encoder, if using a concatenated error correction scheme with RS then Viterbi.
                                    If none, assume the error correction is purely Reed Solomon
    Returns:
        rs_n (int): Maximal Reed Solomon codeword size (a codeword consists of n bytes, k of which are data and n - k of which are parity)
        rs_k (int): Maximal k (number of data bytes) for Reed Solomon codeword
    """
    rs_capacity = data_capacity
    if viterbi_k is not None:
        rs_capacity -= (viterbi_k*2)
        rs_capacity /= 2
    rs_n = int(rs_capacity / 8)
    rs_k = int(np.ceil(raw_signature_size / 8))
    return rs_n, rs_k
    