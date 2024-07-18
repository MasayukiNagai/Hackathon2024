import numpy as np
import gzip

def get_fastas_from_file(fasta_path, uppercase=False):
    """
    Reads a FASTA file and returns the sequences as a NumPy array.

    Parameters
    ----------
    fasta_path : str
        Path to the FASTA file.
    uppercase : bool, optional
        If True, converts sequences to uppercase. Default is False.

    Returns
    -------
    np.ndarray
        Array of sequences from the FASTA file.
    """
    fastas = []
    # headers = []
    seq = None
    header = None
    file_opener = gzip.open if fasta_path.endswith('.gz') else open
    with file_opener(fasta_path, 'rt') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith(">"):
                if seq is not None and header is not None:
                    fastas.append(seq)
                    # headers.append(header)
                seq = ""
                header = line[1:]
            else:
                if seq is not None:
                    seq += line.upper() if uppercase else line
                else:
                    seq = line.upper() if uppercase else line
    fastas.append(seq)
    # headers.append(header)
    return np.array(fastas)


def get_activity_from_file(activity_path):
    """
    Reads a TSV file and returns the data as a NumPy array.

    Parameters
    ----------
    activity_path : str
        Path to the TSV file.

    Returns
    -------
    np.ndarray
        Array of data from the TSV file, skipping the header.
    """
    data = np.genfromtxt(activity_path, delimiter='\t', skip_header=1)
    return data


def convert_one_hot(sequences, alphabet="ACGT", uncertain_N=True):
    """
    Convert a flat array of sequences to one-hot representation.
    **Important**: all letters in `sequences` *must* be contained in `alphabet`,
    and all sequences must have the same length.

    Parameters
    ----------
    sequences : numpy.ndarray of strings
        The array of strings. Should be one-dimensional.
    alphabet : str
        The alphabet of the sequences.

    Returns
    -------
    Numpy array of sequences in one-hot representation. The shape of this array is
    `(len(sequences), len(sequences[0]), len(alphabet))`.

    Examples
    --------
    >>> one_hot(["TGCA"], alphabet="ACGT")
    array([[[0., 0., 0., 1.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.]]])
    """
    A = len(alphabet)
    alphabet += 'N' # to capture non-sense characters

    sequences = np.asanyarray(sequences)
    assert sequences.ndim == 1, "array of sequences must be one-dimensional."

    n_sequences = sequences.shape[0]
    seq_len = len(sequences[0])

    # Unpack strings into 2D array, where each point has one character.
    s = np.zeros((n_sequences, seq_len), dtype="U1")
    for i in range(n_sequences):
        s[i] = list(sequences[i])

    # Make an integer array from the string array.
    pre_onehot = np.zeros(s.shape, dtype=np.uint8)
    for i, letter in enumerate(alphabet):
        # do nothing on 0 because array is initialized with zeros.
        if i:
            pre_onehot[s == letter] = i

    # create one-hot representation
    n_classes = len(alphabet)
    one_hot = np.eye(n_classes)[pre_onehot]

    # remove nonsense character
    one_hot = one_hot[:,:,:A]

    # replace positions with N with 0.25 if true
    if uncertain_N:
        for n,x in enumerate(one_hot):
            index = np.where(np.sum(x, axis=-1) == 0)[0]
            one_hot[n,index,:] = 0.25
    return one_hot
