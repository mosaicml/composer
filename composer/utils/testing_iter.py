import io
from typing import Iterator

def iter_to_stream(iterator: Iterator[bytes], buffer_size: int =io.DEFAULT_BUFFER_SIZE) -> io.BufferedReader:
    """Function to convert iterator of bytes into a file-like binary stream object.

    Original implementation found `here <https://stackoverflow.com/questions/6657820/how-to-convert-an-iterable-to-a-stream/20260030#20260030>`_.
    
    Args:
        iterator (Iterator[bytes]): An iterator over bytes objects
        buffer_size (int): Buffer length of the stream
    
    Returns:
        io.BufferedReader: A buffered binary stream.
    """
    class BytesToStream(io.RawIOBase):
        def __init__(self):
            self.leftover = None
        
        def readinto(self, b):
            try:
                l = len(b) # max bytes to read
                chunk = self.leftover or next(iterator)
                output, self.leftover = chunk[:l], chunk[l:]
                b[:len(output)] = output
                return len(output)
            except StopIteration:
                return 0 #EOF

        def readable(self):
            return True
    
    return io.BufferedReader(BytesToStream(), buffer_size=buffer_size)


test_iterator = iter([b'121341234', b'124235111', b'15235234'])

print(b''.join(test_iterator))

test_iterator = iter([b'121341234', b'124235111', b'15235234'])


a = iter_to_stream(test_iterator, buffer_size=8)
print(a.read())

