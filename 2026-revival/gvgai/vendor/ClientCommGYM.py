import io
import json
import logging
import sys
import os
import random
import tempfile
import shutil

import numpy as np
from PIL import Image


def _raw_frame(sso):
    """RGB frame that followed the message (engine run with -Dgvgai.rawFrames=true), else None."""
    raw = getattr(sso, "imageRawBytes", None)
    if raw is None:
        return None
    shape = (sso.imageHeight, sso.imageWidth)  # greyscale when the server sends one byte per pixel
    if len(raw) != shape[0] * shape[1]:
        shape += (3,)
    return np.frombuffer(raw, np.uint8).reshape(shape)


def _imread(path):
    # scipy.misc.imread was removed in SciPy 1.2
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'))

import subprocess
import argparse

from SerializableStateObservation import SerializableStateObservation, Phase, Observation
from CompetitionParameters import CompetitionParameters
from ElapsedCpuTimer import ElapsedCpuTimer
from IOSocket import IOSocket
from Types import LEARNING_SSO_TYPE


JAVA: str = "java"  # set by the caller to the JDK's java
JAVA_OPTS = []  # set by the caller, e.g. ["-Dgvgai.longDelay=0"]
CLASSES = None  # compiled engine folder, set by the caller (engine/build.py)


class ClientCommGYM:
    """
     * Client communication, set up the socket for a given agent
    """

    def __init__(self, game, version, lvl, pathStr):
        self.tempDir = tempfile.TemporaryDirectory()
        self.addLevel('')   #Level template to be loaded into java

        self.TOKEN_SEP = '#'
        self.io = IOSocket(self.tempDir.name)
        self.sso = SerializableStateObservation()
        #self.agentName = agentName
        self.lastMessageId = 0
        self.LOG = False
        self.player = None
        self.global_ect = None
        self.lastSsoType = LEARNING_SSO_TYPE.JSON
        
        self.sso.Terminal=False

        # pathStr is the engine folder; classes are built by engine/build.py
        buildDir = CLASSES or os.path.join(pathStr, 'build', 'classes')
        gamesDir = os.path.join(pathStr, 'games', '{}_v{}'.format(game, version))
        cmd = [JAVA] + JAVA_OPTS + ["-classpath", buildDir, "tracks.singleLearning.utils.JavaServer", "-game", game, "-gamesDir", gamesDir, "-imgDir", pathStr, "-portNum", str(self.io.port)]
        # stdin/stderr to DEVNULL too: an inherited handle keeps the caller's pipe open after we exit
        self.java = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, cwd=self.tempDir.name)

        self.startComm()
        self.reset(lvl)

    def startComm(self):
        self.io.initBuffers()
        #Reset currently sends initial commucations (which can't handle levels) and then resets
        #This should be split into two functions after the competition (July 18, 2018)
        self.reset(0)

    """
     * Method that perpetually listens for messages from the server.
     * With the use of additional helper methods, this function interprets
     * messages and represents the core response-generation methodology of the agent.
     * @throws IOException
    """

    def step(self,act,repeat=1):
        self.step_send(act, repeat)
        return self.step_recv()

    # Split in two so a caller can send to many servers before waiting on any of them.
    def step_send(self,act,repeat=1):
        self._sent = not self.sso.Terminal
        if self._sent:
            if(act == 0):
                self.act("", repeat)
            else:
                action = self.sso.availableActions[act - 1]
                self.act(action, repeat)

    def step_recv(self):
        #self.sso.phase = Phase.ACT
        if self._sent:
            self.line = self.io.readLine()
            self.line = self.line.rstrip("\r\n")
            self.processLine(self.line)
            
            score = self.reward()
            self.lastScore=self.sso.gameScore
        else:
            score=0
        
        if self.sso.isGameOver==True or self.sso.gameWinner=='PLAYER_WINS' or self.sso.phase == "FINISH" or self.sso.phase=="ABORT" or self.sso.phase=="End":
            self.sso.image = _raw_frame(self.sso) if _raw_frame(self.sso) is not None else _imread(os.path.join(self.tempDir.name, 'gameStateByBytes.png'))
            self.sso.Terminal=True
            #self.lastScore=0
            #Score = self.lastScore
            #self.lastScore=self.sso.gameScore
        else:
            self.sso.Terminal=False
            actions=self.actions()
        
      
        info = {'winner': self.sso.gameWinner, 'actions': self.actions()}  
        return self.sso.image, score, self.sso.Terminal, info

    def reset(self, lvl):
        #flag=True
        #self.line = ''
        self.lastScore=0
        
        if hasattr(self,'line'):
            flag=True
            restart=True
            
            #self.io.writeToServer(self.lastMessageId, "END_TRAINING", self.LOG)
            
            #self.line = self.io.readLine()
            #self.line = self.line.rstrip("\r\n")
            #self.processLine(self.line)
            
            if self.sso.Terminal:
                self.io.writeToServer(self.lastMessageId, str(lvl) + "#" + self.lastSsoType, self.LOG)
            else:
            
                self.io.writeToServer(self.lastMessageId, "END_OVERSPENT", self.LOG)
            
                self.line = self.io.readLine()
                self.line = self.line.rstrip("\r\n")
                self.processLine(self.line)
            
                self.io.writeToServer(self.lastMessageId, str(lvl) + "#" + self.lastSsoType, self.LOG)

        else:
            restart=True
            flag=True
            #self.startComm()
            self.line = ''


        while flag:
            if restart:
                self.line = self.io.readLine()
                self.line = self.line.rstrip("\r\n")
                self.processLine(self.line)
            else:
                self.line=''

            if self.sso.phase == Phase.START:
                self.start()


            elif self.sso.phase == "INIT":
                self.sso.phase = Phase.INIT
                self.init()

            elif self.sso.phase == "ACT":
                flag=False                
                
                
                for i in range(1):
                    self.act(0)
                    self.line = self.io.readLine()
                    self.line = self.line.rstrip("\r\n")
                    self.processLine(self.line)

                #print(self.sso.phase)
                #print(self.sso.isGameOver)
                #print(dir(self.sso))

                if(self.sso.isGameOver==True or self.sso.gameWinner=='WINNER' or self.sso.phase == "FINISH" or self.sso.phase == "End"):
                    
                    self.sso.image = _raw_frame(self.sso) if _raw_frame(self.sso) is not None else _imread(os.path.join(self.tempDir.name, 'gameStateByBytes.png'))
                    self.sso.Terminal=True
                    self.lastScore=0
                else:
                    self.sso.Terminal=False
                

        return self.sso.image

    # def reward(self):
    #     scoreDelta = self.sso.gameScore-self.lastScore
    #     if(self.sso.gameWinner=='WINNER' or scoreDelta > 0):
    #         return 1
    #     elif(self.sso.isGameOver or scoreDelta < 0):
    #         return -1
    #     else:
    #         return 0

    def reward(self):
        scoreDelta = self.sso.gameScore-self.lastScore
        return scoreDelta

    def actions(self):
        nil = ["ACTION_NIL"]
        return nil + self.sso.availableActions

    def as_sso(self, d):
        self.sso.__dict__.update(d)
        return self.sso

    def parse_json(self, input):
        parsed_input = json.loads(input)
        self.sso.__dict__.update(parsed_input)
        if parsed_input.get('observationGrid'):
            self.sso.observationGrid = [[[None for j in range(self.sso.observationGridMaxCol)]
                                         for i in range(self.sso.observationGridMaxRow)]
                                        for k in range(self.sso.observationGridNum)]
            for i in range(self.sso.observationGridNum):
                for j in range(len(parsed_input['observationGrid'][i])):
                    for k in range(len(parsed_input['observationGrid'][i][j])):
                        self.sso.observationGrid[i][j][k] = Observation(parsed_input['observationGrid'][i][j][k])

        if parsed_input.get('NPCPositions'):
            self.sso.NPCPositions = [[None for j in
                                      range(self.sso.NPCPositionsMaxRow)] for i in
                                     range(self.sso.NPCPositionsNum)]
            for i in range(self.sso.NPCPositionsNum):
                for j in range(len(parsed_input['NPCPositions'][i])):
                    self.sso.NPCPositions[i][j] = Observation(parsed_input['NPCPositions'][i][j])

        if parsed_input.get('immovablePositions'):
            self.sso.immovablePositions = [[None for j in
                                            range(self.sso.immovablePositionsMaxRow)] for i in
                                           range(self.sso.immovablePositionsNum)]
            for i in range(self.sso.immovablePositionsNum):
                for j in range(len(parsed_input['immovablePositions'][i])):
                    self.sso.immovablePositions[i][j] = Observation(parsed_input['immovablePositions'][i][j])

        if parsed_input.get('movablePositions'):
            self.sso.movablePositions = [[None for j in
                                          range(self.sso.movablePositionsMaxRow)] for i in
                                         range(self.sso.movablePositionsNum)]
            for i in range(self.sso.movablePositionsNum):
                for j in range(len(parsed_input['movablePositions'][i])):
                    self.sso.movablePositions[i][j] = Observation(parsed_input['movablePositions'][i][j])

        if parsed_input.get('resourcesPositions'):
            self.sso.resourcesPositions = [[None for j in
                                            range(self.sso.resourcesPositionsMaxRow)] for i in
                                           range(self.sso.resourcesPositionsNum)]
            for i in range(self.sso.resourcesPositionsNum):
                for j in range(len(parsed_input['resourcesPositions'][i])):
                    self.sso.resourcesPositions[i][j] = Observation(parsed_input['resourcesPositions'][i][j])

        if parsed_input.get('portalsPositions'):
            self.sso.portalsPositions = [[None for j in
                                          range(self.sso.portalsPositionsMaxRow)] for i in
                                         range(self.sso.portalsPositionsNum)]
            for i in range(self.sso.portalsPositionsNum):
                for j in range(len(parsed_input['portalsPositions'][i])):
                    self.sso.portalsPositions[i][j] = Observation(parsed_input['portalsPositions'][i][j])

        if parsed_input.get('fromAvatarSpritesPositions'):
            self.sso.fromAvatarSpritesPositions = [[None for j in
                                                    range(self.sso.fromAvatarSpritesPositionsMaxRow)] for i in
                                                   range(self.sso.fromAvatarSpritesPositionsNum)]
            for i in range(self.sso.fromAvatarSpritesPositionsNum):
                for j in range(len(parsed_input['fromAvatarSpritesPositions'][i])):
                    self.sso.fromAvatarSpritesPositions[i][j] = Observation(parsed_input['fromAvatarSpritesPositions'][i][j])


    """
     * Method that interprets the received messages from the server's side.
     * A message can either be a string (in the case of initialization), or
     * a json object containing an encapsulated state observation.
     * This method deserializes the json object into a local state observation
     * instance.
     * @param msg Message received from server to be interpreted.
     * @throws IOException
    """

    def processLine(self, msg):
        try:
            if msg is None:
                print ("Message is null")
                return

            message = msg.split(self.TOKEN_SEP)
            if len(message) < 2:
                print ("Message not complete")
                return

            self.lastMessageId = message[0]
            js = message[1]

            self.sso = SerializableStateObservation()
            if js == "START":
                self.sso.phase = Phase.START
            elif js == "FINISH":
                self.sso.phase = Phase.FINISH
            else:
                js.replace('"', '')
                self.parse_json(js)
                if getattr(self.sso, "imageBytes", 0):
                    self.sso.imageRawBytes = self.io.read_exact(self.sso.imageBytes)
                # self.sso = json.loads(js, object_hook=self.as_sso)
            if self.sso.phase == "ACT":
                if(self.lastSsoType == LEARNING_SSO_TYPE.IMAGE or self.lastSsoType == "IMAGE" \
                        or self.lastSsoType == LEARNING_SSO_TYPE.BOTH or self.lastSsoType == "BOTH"):
                    raw = _raw_frame(self.sso)
                    if raw is not None:
                        self.sso.image = raw
                    elif(self.sso.imageArray):
                        # Decode in memory; upstream wrote the PNG to disk and read it back each tick.
                        png = np.asarray(self.sso.imageArray, dtype=np.int8).view(np.uint8).tobytes()
                        with Image.open(io.BytesIO(png)) as im:
                            self.sso.image = np.asarray(im.convert('RGB'))

        except Exception as e:
            logging.exception(e)
            print("Line processing [FAILED]")
            #traceback.print_exc()
            sys.exit()

    """
     * Manages the start of the communication. It starts the whole process, and sets up the timer for the whole run.
    """

    def start(self):
        self.global_ect = ElapsedCpuTimer()
        self.global_ect.setMaxTimeMillis(CompetitionParameters.TOTAL_LEARNING_TIME)
        ect = ElapsedCpuTimer()
        ect.setMaxTimeMillis(CompetitionParameters.START_TIME)
        #self.startAgent()

        if ect.exceededMaxTime():
            self.io.writeToServer(self.lastMessageId, "START_FAILED", self.LOG)
        else:
            self.io.writeToServer(self.lastMessageId, "START_DONE" + "#" + self.lastSsoType, self.LOG)


    def init(self):
        ect = ElapsedCpuTimer()
        ect.setMaxTimeMillis(CompetitionParameters.INITIALIZATION_TIME)
        #self.player.init(self.sso, ect.copy())
        #self.lastSsoType = self.player.lastSsoType
        self.lastSsoType = LEARNING_SSO_TYPE.IMAGE
        actions=self.actions()

        if ect.exceededMaxTime():
            self.io.writeToServer(self.lastMessageId, "INIT_FAILED", self.LOG)
        else:
            self.io.writeToServer(self.lastMessageId, "INIT_DONE" + "#" + self.lastSsoType, self.LOG)

    """
     * Manages the action request for an agent. The agent is requested for an action,
     * which is sent back to the server
    """

    def act(self,action,repeat=1):
        ect = ElapsedCpuTimer()
        ect.setMaxTimeMillis(CompetitionParameters.ACTION_TIME)
        #action = str(self.player.act(self.sso, ect.copy()))
        if (not action) or (action == ""):
            action = "ACTION_NIL"
        #self.lastSsoType = self.player.lastSsoType
        self.lastSsoType = LEARNING_SSO_TYPE.IMAGE


        if ect.exceededMaxTime():
            if ect.elapsedNanos() > CompetitionParameters.ACTION_TIME_DISQ*1000000:
                self.io.writeToServer(self.lastMessageId, "END_OVERSPENT", self.LOG)
            else:
                self.io.writeToServer(self.lastMessageId, "ACTION_NIL" + "#" + self.lastSsoType, self.LOG)
        else:
            # optional repeat count: the server plays the action that many ticks and sends only the last frame
            extra = "#" + str(repeat) if repeat > 1 else ""
            self.io.writeToServer(self.lastMessageId, action + "#" + self.lastSsoType + extra, self.LOG)

    def addLevel(self, path):
        lvlName = os.path.join(self.tempDir.name, 'game_lvl5.txt')
        if(path == ''):
            open(lvlName, 'w+').close()
        else:
            shutil.copyfile(path, lvlName)

    def __del__(self):
        try:
            self.java.kill()
        except:
            pass
