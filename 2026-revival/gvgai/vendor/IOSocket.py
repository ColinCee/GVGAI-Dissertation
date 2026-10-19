import logging
import os
import socket
import sys
import traceback
import time

from CompetitionParameters import CompetitionParameters


class IOSocket:
    """
     * Socket for communication
    """

    def __init__(self, tmpDir):
        self.BUFF_SIZE = 8192
        self.END = '\n'
        self.TOKEN_SEP = '#'
        self.hostname, self.port = self.getOpenAddress()
        self.connected = False
        self._buf = bytearray()
        self.socket = None
        self.logfilename = os.path.normpath(os.path.join(tmpDir, "logs", "clientLog.txt"))

        os.makedirs(os.path.join(tmpDir, 'logs'), exist_ok=True)
        self.logfile = open(self.logfilename, "a")

    def initBuffers(self):
        pass  # print ("Connecting to host " + str(self.hostname) + " at port " + str(self.port) + " ...")
        while not self.connected:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.hostname, self.port))
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.connected = True
                pass  # print ("Client connected to server [OK]")
            except Exception as e:
                time.sleep(1)
                # logging.exception(e)
                # print("Client connected to server [FAILED]")
                # traceback.print_exc()
                # sys.exit()

    def writeToFile(self, line):
        sys.stdout.write(line + os.linesep)
        self.logfile.write(line + os.linesep)
        sys.stdout.flush()
        self.logfile.flush()

    def writeToServer(self, messageId, line, log):
        msg = str(messageId) + self.TOKEN_SEP + line + "\n"
        try:
            self.socket.send(msg.encode())
            if log:
                self.writeToFile(msg.strip('\n'))
        except Exception as e:
            logging.exception(e)
            print ("Write " + self.logfilename + " to server [FAILED]")
            traceback.print_exc()
            sys.exit()

    def readLine(self):
        try:
            msg = self.recv_end()
            return msg
        except Exception as e:
            logging.exception(e)
            print ("Read from server [FAILED]")
            traceback.print_exc()
            sys.exit()
    
    def shutDown(self):
        self.socket.shutdown(0)

    def getOpenAddress(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("localhost",0))
        address = s.getsockname()
        s.close()
        return address
    
    def recv_end(self):
        # Buffered: a binary frame can follow the line (see read_exact), so keep what comes after the newline.
        buf = self._buf
        while True:
            i = buf.find(b"\n")
            if i >= 0:
                line = bytes(buf[:i])
                del buf[: i + 1]
                return line.decode()
            chunk = self.socket.recv(1 << 20)
            if not chunk:  # the server is gone: fail instead of spinning forever
                raise ConnectionError("game server closed the connection")
            buf += chunk

    def read_exact(self, n):
        buf = self._buf
        while len(buf) < n:
            chunk = self.socket.recv(max(1 << 20, n - len(buf)))
            if not chunk:
                raise ConnectionError("game server closed the connection")
            buf += chunk
        out = bytes(buf[:n])
        del buf[:n]
        return out
