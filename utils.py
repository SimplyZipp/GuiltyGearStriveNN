import time
import numpy as np
import cv2
from mss import mss
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SimpleTimer:
    """
    Simple, blocking timer
    """
    def __init__(self, interval, verbose=False):
        self.interval = interval
        self.time = 0
        self.verbose = verbose

    def start(self):
        self.time = time.time()

    def wait(self):
        remaining = self.interval - (time.time()-self.time)
        if remaining > 0:
            time.sleep(remaining)
        if self.verbose:
            print(f'Given interval is {self.interval}, time taken was {time.time()-self.time}')

    def wait_and_continue(self):
        self.wait()
        self.start()


class FrameGrabber:
    def __init__(self, size=(800, 450), bounds=None):
        if bounds is None:
            bounds = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        self.size = size
        self.bounds = bounds
        self.sct = mss()

        # Maybe normalize the image somehow?
        self.transform = A.Compose(
            [A.Resize(*self.size, interpolation=cv2.INTER_AREA),
             ToTensorV2()]
        )

    def frame(self):
        """
        Get a frame from the screen, resizing it to 'size' and making it grayscale
        :return: the modified frame image as a tensor
        """

        sct_img = self.sct.grab(self.bounds)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self.transform(image=img)['image']
        return img

    def color_img(self):
        sct_img = self.sct.grab(self.bounds)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    
    
#here's the menu navigation and controller part. Always start the looop at the side select screen
from pymem import *
from pymem.process import *
import vgamepad as vg
from time import sleep


class GameManager:
    def __init__(self):
        self.startpixels = ((771, 405), (429, 540), (421, 745), (444, 452), (366, 277), (531, 317), (1320, 416), (1300, 639), (953, 285), (906, 599), (1102, 711), (1404, 674), (379, 303), (1025, 184), (888, 693), (1129, 698), (440, 434), (317, 226), (1539, 216), (1606, 855), (1240, 724))
        self.antipixel = ((1849, 364), (1872, 702))

        self.d = {1: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_SOUTHWEST,
                  2: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_SOUTH,
                  3: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_SOUTHEAST,
                  6: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_EAST,
                  9: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NORTHEAST,
                  8: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NORTH,
                  7: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NORTHWEST,
                  4: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_WEST,
                  5: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NONE}

        self.buttons = buttons = {"punch": vg.DS4_BUTTONS.DS4_BUTTON_SQUARE,
                                  "kick": vg.DS4_BUTTONS.DS4_BUTTON_CROSS, #also acts as confirm
                                  "slash": vg.DS4_BUTTONS.DS4_BUTTON_TRIANGLE,
                                  "hslash": vg.DS4_BUTTONS.DS4_BUTTON_CIRCLE, #also acts as exit
                                  "dust": vg.DS4_BUTTONS.DS4_BUTTON_TRIGGER_RIGHT,
                                  "dash": vg.DS4_BUTTONS.DS4_BUTTON_THUMB_LEFT}

        self.pm = Pymem('GGST-Win64-Shipping.exe')
        self.gamepad = vg.VDS4Gamepad()

    def dpad(self, num):
        self.gamepad.directional_pad(direction=self.d[num])
        self.gamepad.update()

    def button(self, inp):
        self.gamepad.press_button(button=self.buttons[inp])
        self.gamepad.update()

    def buttonr(self, inp):
        self.gamepad.release_button(button=self.buttons[inp[:-1]])
        self.gamepad.update()

    def GetPtrAddr(self, base, offsets):
        addr = self.pm.read_longlong(base)
        for i in offsets:
            if i != offsets[-1]:
                addr = self.pm.read_longlong(addr + i)
        return addr + offsets[-1]

    def MatchStart(self, pixels):  # use the color image from the main loop to determine if the match is starting
        count = 0
        for i in self.startpixels:
            if (pixels[i[1], i[0]] == (255, 255, 255)).all():
                count = count + 1
            else:
                break
        for i in self.antipixel:
            if (pixels[i[1], i[0]] == (255, 255, 255)).all():
                break
            else:
                count = count + 1
        if (count == len(self.startpixels) + len(self.antipixel)):
            return True
        else:
            return False

    def MatchEnd(self):  # uses the HP values in memory to determine if the round has ended
        p1HP = self.pm.read_int(self.GetPtrAddr(self.pm.base_address + 0x0505E6B8, offsets=[0x138, 0x0, 0x6E0, 0x688, 0x1178]))
        p2HP = self.pm.read_int(self.GetPtrAddr(self.pm.base_address + 0x0505E6B8, offsets=[0x138, 0x8, 0x6E0, 0x688, 0x1178]))
        if min(p1HP, p2HP) == 0:
            return True
        else:
            return False

    def initloop(self, charanum, enemytype, enemynum):  # for selecting a side and picking a character
        if enemytype != "cpu" and enemytype != "enemy":
            print("wrong enemy type")
            return

        gamepad = self.gamepad
        pm = self.pm

        players = np.arange(1, 5)
        addresses = [0x0, 0x4406D8C, 0x4406D94, 0x4406D9C, 0x4406DA4]
        position = {}
        player1taken = False
        player2taken = False
        side = 0
        time.sleep(2)
        gamepad.reset()
        gamepad.update()
        for i in players:
            position[i] = pm.read_int(pm.base_address + addresses[i])
            if position[i] == 1:
                player1taken = True
            if position[i] == 2:
                player2taken = True
        if player1taken == False:
            self.dpad(4)
            time.sleep(.1)
            self.dpad(5)
            side = 1
        elif player2taken == False:
            self.dpad(6)
            time.sleep(.1)
            self.dpad(5)
            side = 2
        else:
            print("both sides taken")
            return
        gamepad.reset()
        sideselect = pm.read_int(pm.base_address + 0x4406D88)
        while sideselect == True or sideselect == 2:  # tutorial side select is 2 for some reason
            time.sleep(3)
            self.button("kick")
            time.sleep(.1)
            self.buttonr("kickr")
            sideselect = pm.read_int(pm.base_address + 0x4406D88)
        time.sleep(1)
        if enemytype == "cpu":
            self.button("kick")
            time.sleep(.2)
            self.buttonr("kickr")
        time.sleep(2)
        # close the option select

        time.sleep(.1)
        self.button("slash")
        time.sleep(.1)
        self.buttonr("slashr")

        self.dpad(2)
        time.sleep(5)  # open button settings and go to the bottom

        self.dpad(8)
        time.sleep(.05)
        self.dpad(5)
        time.sleep(.05)
        self.dpad(8)  # hover over default settings
        time.sleep(.05)
        self.dpad(5)
        self.button("kick")
        time.sleep(.1)
        self.buttonr("kickr")
        self.button("hslash")
        time.sleep(.1)
        self.buttonr("hslashr")  # select default settings and return to character select
        self.dpad(2)

        self.character_select(charanum, side)

        if enemytype == "cpu":
            self.character_select(enemynum, (side % 2) + 1)
        # move onto the main loop


    def character_select(self, charanum, side):
        pm = self.pm

        character = 0
        while 1:  # this loop keeps on moving the selector right one until its on a character it knows
            if side == 1:
                character = pm.read_int(
                    self.GetPtrAddr(pm.base_address + 0x0505E6B8, offsets=[0x168, 0x20, 0x98, 0x1D0, 0x29704]))
            else:
                character = pm.read_int(
                    self.GetPtrAddr(pm.base_address + 0x0505E6B8, offsets=[0x208, 0x5E8, 0x658, 0x2E8, 0x29704]))
            if (character == 1 or character == 3):
                break
            else:
                self.dpad(6)
                time.sleep(.05)
                self.dpad(5)
                time.sleep(.5)
        top = np.array([19, 16, 11, 7, 4, 0, 1, 2, 8, 14, 17])

        # Bottom is not used?
        bottom = np.array([18, 13, 10, 6, 3, 33, 5, 9, 12, 15])  # characters located on teh top and bottom

        if min((charanum - top) ** 2) == 0:  # determines which row the pointer is on
            row = 1
        else:
            row = 0

        if (row == 1 and character == 1) or (
                row == 0 and character == 3):  # makes sure the selector is on the right row
            pass
        else:
            self.dpad(2)
            time.sleep(.1)
            self.dpad(5)
        character = 0
        while 1:  # moves the selector right again, but this time doesn't stop until its on the right character
            if side == 1:
                character = pm.read_int(
                    self.GetPtrAddr(pm.base_address + 0x0505E6B8, offsets=[0x168, 0x20, 0x98, 0x1D0, 0x29704]))
            else:
                character = pm.read_int(
                    self.GetPtrAddr(pm.base_address + 0x0505E6B8, offsets=[0x208, 0x5E8, 0x658, 0x2E8, 0x29704]))
            if (character == charanum):
                break
            else:
                self.dpad(6)
                time.sleep(.05)
                self.dpad(5)
                time.sleep(1)
        self.button("kick")  # select the character
        time.sleep(.1)
        self.buttonr("kickr")
        self.dpad(6)  # select the color of the character
        time.sleep(.1)
        self.dpad(5)
        time.sleep(.1)
        if side == 1:
            self.dpad(6)
            time.sleep(.1)
            self.dpad(5)
        self.button("kick")
        time.sleep(.1)
        self.buttonr("kickr")
        return
        # return to the main loop

    def action(self, prev_move, prev_button, move, new_button):
        frame = .017
        holdbutton = True

        if prev_button > 10 and prev_button != 13:  # cases where the last button should be released
            holdbutton = False
        elif prev_button < 5 and new_button - prev_button != 5:
            holdbutton = False
        elif prev_button != new_button:
            holdbutton = False

        if prev_move == 20:  # if no previous moves have been taken
            pass
        elif (holdbutton == False):  # if you change buttons or no longer hold a button
            if prev_button == 0 or prev_button == 5:
                self.buttonr("punchr")
            elif prev_button == 1 or prev_button == 6:
                self.buttonr("kickr")
            elif prev_button == 2 or prev_button == 7:
                self.buttonr("slashr")
            elif prev_button == 3 or prev_button == 8:
                self.buttonr("hslashr")
            elif prev_button == 4 or prev_button == 9:
                self.buttonr("dustr")
            elif prev_button == 13:
                self.buttonr("dashr")
            elif prev_button == 10:
                self.buttonr("punchr")
                self.buttonr("kickr")
        if (new_button > 4 and new_button <= 10) or new_button == 13:  # if system says to hold a button
            if new_button == 5:
                self.button("punch")
            elif new_button == 6:
                self.button("kick")
            elif new_button == 7:
                self.button("slash")
            elif new_button == 8:
                self.button("hslash")
            elif new_button == 9:
                self.button("dust")
            elif new_button == 13:
                self.button("dash")
            elif new_button == 10:
                self.button("punch")
                self.button("kick")
        if move == 0:  # cases for movement
            self.dpad(6)
            sleep(frame * 4)
        elif move == 1:
            self.dpad(4)
            sleep(frame * 4)
        elif move == 2:
            self.dpad(8)
            sleep(frame * 4)
        elif move == 3:
            self.dpad(2)
            sleep(frame * 4)
        elif move == 4:
            self.dpad(2)
            sleep(frame)
            self.dpad(3)
            sleep(frame)
            self.dpad(6)
            sleep(frame * 2)
        elif move == 5:
            self.dpad(2)
            sleep(frame)
            self.dpad(1)
            sleep(frame)
            self.dpad(4)
            sleep(frame * 2)
        elif move == 6:
            self.dpad(6)
            sleep(frame)
            self.dpad(2)
            sleep(frame)
            self.dpad(3)
            sleep(frame * 2)
        elif move == 7:
            self.dpad(4)
            sleep(frame)
            self.dpad(2)
            sleep(frame)
            self.dpad(3)
            sleep(frame * 2)
        elif move == 8:
            self.dpad(4)
            sleep(frame)
            self.dpad(1)
            sleep(frame)
            self.dpad(2)
            sleep(frame)
            self.dpad(6)
            sleep(frame)
        elif move == 9:
            self.dpad(6)
            sleep(frame)
            self.dpad(2)
            sleep(frame)
            self.dpad(4)
            sleep(frame)
        elif move == 10:
            self.dpad(1)
            sleep(frame * 4)
        elif move == 11:
            self.dpad(3)
            sleep(frame * 3)
        elif move == 12:
            self.dpad(7)
            sleep(frame * 3)
        elif move == 13:
            self.dpad(9)
            sleep(frame * 4)
        elif move == 14:
            self.dpad(5)
            sleep(frame * 4)

        if new_button == 0:  # cases for holding the button late
            self.button("punch")
        if new_button == 1:
            self.button("kick")
        if new_button == 2:
            self.button("slash")
        if new_button == 3:
            self.button("hslash")
        if new_button == 4:
            self.button("dust")
        if new_button == 10:
            self.button("punch")
            self.button("kick")
        if new_button == 11:
            self.button("punch")
            self.button("kick")
            self.button("slash")
        if new_button == 12:
            self.button("dust")
            self.button("punch")
        if new_button == 13:
            self.button("dash")
        sleep(frame)

    def get_reward(self, prev_stats, position: str):
        stats = self.getcurrentstats()
        return self._getReward(*prev_stats, *stats, position), stats

    def getcurrentstats(self):
        pm = self.pm
        # these are the values at the current time t, unless it works differently on different systems, don't worry about changing
        p2HP = pm.read_int(self.GetPtrAddr(pm.base_address + 0x0505E6B8, offsets=[0x138, 0x8, 0x6E0, 0x688, 0x1178]))
        p1HP = pm.read_int(self.GetPtrAddr(pm.base_address + 0x0505E6B8, offsets=[0x138, 0x0, 0x6E0, 0x688, 0x1178]))
        p1RISC = pm.read_int(self.GetPtrAddr(pm.base_address + 0x0505E6B8, offsets=[0x130, 0x8, 0x688, 0xC764]))
        p2RISC = pm.read_int(self.GetPtrAddr(pm.base_address + 0x0505E6B8, offsets=[0x138, 0x8, 0x2A8, 0xC764]))
        # print("health P1:" + str(p1HP))
        # print("health P2: " + str(p2HP))
        # print("RISC P1: " + str(p1RISC))
        # print("RISC P2: " + str(p2RISC))
        return p1HP, p2HP, p1RISC, p2RISC

    def _getReward(self, prev_p1HP, prev_p2HP, prev_p1RISC, prev_p2RISC, p1HP, p2HP, p1RISC, p2RISC, position):
        if (position == "Player 1"):  # change to whatever indicator we use to say whether the AI is player 1 or two
            mult = 1
        else:
            mult = -1
        Reward = 0
        # current reward function, can adjust values and add more cases to help improve the behavior of the actor
        Reward -= 20 * (p1HP == 0) * mult
        Reward += 20 * (p2HP == 0) * mult  # + 20 if you win a round, -20 if you lose

        Reward += (prev_p1HP - p1HP) / 42 * mult
        Reward -= (prev_p2HP - p2HP) / 42 * mult  # +1 point for every 10% damage dealt, max HP is 420

        Reward -= max(p1RISC - prev_p1RISC,
                      0) / 12800 * mult  # +1 point per bar of enemy RISC filled. Max RISC value is 12800.
        Reward += max(p2RISC - prev_p2RISC,
                      0) / 12800 * mult  # if RISC goes below zero, then the enemy is being combo'd and not part of this case

        Reward -= 2 * (p1RISC == 12800) * mult
        Reward += 2 * (
                    p2RISC == 12800) * mult  # +2 if enemies RISC is full.if the RISC bar is full, really bad things can happen

        Reward -= (p1RISC > prev_p1RISC) * (prev_p1RISC == 0) / 4
        Reward += (p2RISC > prev_p2RISC) * (prev_p2RISC == 0) / 4  # +.25 for winning neutral or okizeme

        Reward += ((prev_p1RISC - p1RISC) * (prev_p1HP == p1HP) * (
                    abs(prev_p1RISC - p1RISC) < 1500)) / 12800  # gets some reward back for allowing RISC to lower without taking damage
        Reward -= ((prev_p2RISC - p2RISC) * (prev_p2HP == p2HP) * (
                    abs(prev_p2RISC - p2RISC) < 1500)) / 12800  # if the RISC drops too much, then you likely just got back up after a combo, so no reward should be given

        # the expected reward should realistically stay between negative 20 and 20
        return Reward


# startpixels = ((771, 405), (429, 540), (421, 745), (444, 452), (366, 277), (531, 317), (1320, 416), (1300, 639), (953, 285), (906, 599), (1102, 711), (1404, 674), (379, 303), (1025, 184), (888, 693), (1129, 698), (440, 434), (317, 226), (1539, 216), (1606, 855), (1240, 724))
# antipixel = ((1849, 364), (1872, 702))
# def dpad(num):
#     d = {1: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_SOUTHWEST,
#     2: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_SOUTH,
#     3: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_SOUTHEAST,
#     6: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_EAST,
#     9: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NORTHEAST,
#     8: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NORTH,
#     7: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NORTHWEST,
#     4: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_WEST,
#     5: vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NONE}
#     gamepad.directional_pad(direction = d[num])
#     gamepad.update()
# def button(inp):
#     buttons = {"punch": vg.DS4_BUTTONS.DS4_BUTTON_SQUARE,
#     "kick": vg.DS4_BUTTONS.DS4_BUTTON_CROSS, #also acts as confirm
#     "slash": vg.DS4_BUTTONS.DS4_BUTTON_TRIANGLE,
#     "hslash": vg.DS4_BUTTONS.DS4_BUTTON_CIRCLE, #also acts as exit
#     "dust": vg.DS4_BUTTONS.DS4_BUTTON_TRIGGER_RIGHT,
#     "dash": vg.DS4_BUTTONS.DS4_BUTTON_THUMB_LEFT}
#     gamepad.press_button(button = buttons[inp])
#     gamepad.update()
# def buttonr(inp):
#     buttons = {"punchr":  vg.DS4_BUTTONS.DS4_BUTTON_SQUARE,
#     "kickr": vg.DS4_BUTTONS.DS4_BUTTON_CROSS,
#     "slashr": vg.DS4_BUTTONS.DS4_BUTTON_TRIANGLE,
#     "hslashr": vg.DS4_BUTTONS.DS4_BUTTON_CIRCLE,
#     "dustr": vg.DS4_BUTTONS.DS4_BUTTON_TRIGGER_RIGHT,
#     "dashr": vg.DS4_BUTTONS.DS4_BUTTON_THUMB_LEFT}
#     gamepad.release_button(button = buttons[inp])
#     gamepad.update()
#
# pm = Pymem('GGST-Win64-Shipping.exe')
#
# #from stack overflow
# def GetPtrAddr(base, offsets):
#     addr = pm.read_longlong(base)
#     for i in offsets:
#         if i != offsets[-1]:
#             addr = pm.read_longlong(addr + i)
#     return addr + offsets[-1]
#
# def MatchStart(pixels): #use the color image from the main loop to determine if the match is starting
#     count = 0
#     for i in startpixels:
#         if(pixels[i[0],i[1]] == (255, 255, 255)):
#             count = count + 1
#         else:
#             break
#     for i in antipixel:
#         if(pixels[i[0], i[1]] == (255, 255, 255)):
#             break
#         else:
#             count = count + 1
#     if(count == len(startpixels) + len(antipixel)):
#         return True
#     else:
#         return False
# def MatchEnd(): #uses the HP values in memory to determine if the round has ended
#     p1HP = pm.read_int(GetPtrAddr(pm.base_address + 0x0505E6B8, offsets = [0x138, 0x0, 0x6E0, 0x688, 0x1178]))
#     p2HP = pm.read_int(GetPtrAddr(pm.base_address + 0x0505E6B8, offsets = [0x138, 0x8, 0x6E0, 0x688, 0x1178]))
#     if min(p1HP, p2HP) == 0:
#         return True
#     else:
#         return False
#
# def initloop(charanum, enemytype, enemynum): #for selecting a side and picking a character
#     if enemytype != "cpu" and enemytype!= "enemy":
#         print("wrong enemy type")
#         return
#     global gamepad
#     gamepad = vg.VDS4Gamepad()
#     players = np.arange(1, 5)
#     addresses = [0x0, 0x4406D8C, 0x4406D94, 0x4406D9C, 0x4406DA4]
#     position = {}
#     player1taken = False
#     player2taken = False
#     side = 0
#     time.sleep(5)
#     gamepad.reset()
#     gamepad.update()
#     for i in players:
#         position[i] = pm.read_int(pm.base_address + addresses[i])
#         if position[i] == 1:
#             player1taken = True
#         if position[i] == 2:
#             player2taken = True
#     if player1taken == False:
#         dpad(4)
#         time.sleep(.1)
#         dpad(5)
#         side = 1
#     elif player2taken == False:
#         dpad(6)
#         time.sleep(.1)
#         dpad(5)
#         side = 2
#     else:
#         print("both sides taken")
#         return
#     gamepad.reset()
#     sideselect = pm.read_int(pm.base_address + 0x4406D88)
#     while sideselect == True or sideselect == 2: #tutorial side select is 2 for some reason
#         time.sleep(3)
#         button("kick")
#         time.sleep(.1)
#         buttonr("kickr")
#         sideselect = pm.read_int(pm.base_address + 0x4406D88)
#     time.sleep(1)
#     if enemytype == "cpu":
#         button("kick")
#         time.sleep(.2)
#         buttonr("kickr")
#     time.sleep(2)
#           #close the option select
#     button("slash")
#     time.sleep(.1)
#     buttonr("slashr")
#     dpad(2)
#     time.sleep(5) #open button settings and go to the bottom
#     dpad(8)
#     time.sleep(.05)
#     dpad(5)
#     time.sleep(.05)
#     dpad(8)     #hover over default settings
#     time.sleep(.05)
#     dpad(5)
#     button("kick")
#     time.sleep(.1)
#     buttonr("kickr")
#     button("hslash")
#     time.sleep(.1)
#     buttonr("hslashr")     #select default settings and return to character select
#     dpad(2)
#
#     character_select(charanum, side)
#
#     if enemytype == "cpu":
#         character_select(enemynum, (side % 2) + 1 )
#     #move onto the main loop
#
#
# def character_select(charanum, side):
#     character = 0
#     while 1: #this loop keeps on moving the selector right one until its on a character it knows
#         if side == 1:
#             character = pm.read_int(GetPtrAddr(pm.base_address + 0x05072918, offsets=[0x118, 0x370, 0x410, 0x20, 0x29704]))
#         else:
#             character = pm.read_int(GetPtrAddr(pm.base_address + 0x04899B78, offsets=[0x358, 0x190, 0x310, 0xA8, 0x29704]))
#         if(character == 1 or character == 3):
#             break
#         else:
#             dpad(6)
#             time.sleep(.05)
#             dpad(5)
#             time.sleep(.5)
#     top = np.array([19, 16, 11, 7, 4, 0, 1, 2, 8, 14, 17])
#     bottom = np.array([18, 13, 10, 6, 3, 33, 5, 9, 12, 15]) #characters located on teh top and bottom
#     if min((charanum - top) ** 2) == 0: #determines which row the pointer is on
#         row = 1
#     else:
#         row = 0
#
#     if (row == 1 and character == 1) or (row == 0 and character == 3): #makes sure the selector is on the right row
#         pass
#     else:
#         dpad(2)
#         time.sleep(.1)
#         dpad(5)
#     character = 0
#     while 1: #moves the selector right again, but this time doesn't stop until its on the right character
#         if side == 1:
#             character = pm.read_int(GetPtrAddr(pm.base_address + 0x05072918, offsets=[0x118, 0x370, 0x410, 0x20, 0x29704]))
#         else:
#             character = pm.read_int(GetPtrAddr(pm.base_address + 0x04899B78, offsets=[0x358, 0x190, 0x310, 0xA8, 0x29704]))
#         if(character == charanum):
#             break
#         else:
#             dpad(6)
#             time.sleep(.05)
#             dpad(5)
#             time.sleep(1)
#     button("kick") #select the character
#     time.sleep(.1)
#     buttonr("kickr")
#     dpad(6) #select the color of the character
#     time.sleep(.1)
#     dpad(5)
#     time.sleep(.1)
#     if side == 1:
#         dpad(6)
#         time.sleep(.1)
#         dpad(5)
#     button("kick")
#     time.sleep(.1)
#     buttonr("kickr")
#     return
#     #return to the main loop
#
# #initloop(18, "cpu", 18)
#
#
# def action(prev_move, prev_button, move, new_button):
#     frame = .017
#     holdbutton = True
#
#     if prev_button > 10 and prev_button != 13: #cases where the last button should be released
#         holdbutton = False
#     elif prev_button < 5 and new_button - prev_button != 5:
#         holdbutton = False
#     elif prev_button != new_button:
#         holdbutton = False
#
#     if prev_move == 20: #if no previous moves have been taken
#         pass
#     elif (holdbutton == False): #if you change buttons or no longer hold a button
#         if prev_button == 0 or prev_button == 5:
#             buttonr("punchr")
#         elif prev_button == 1 or prev_button == 6:
#             buttonr("kickr")
#         elif prev_button == 2 or prev_button == 7:
#             buttonr("slashr")
#         elif prev_button == 3 or prev_button == 8:
#             buttonr("hslashr")
#         elif prev_button == 4 or prev_button == 9:
#             buttonr("dustr")
#         elif prev_button == 13:
#             buttonr("dashr")
#         elif prev_button == 10:
#             buttonr("punchr")
#             buttonr("kickr")
#     if (new_button > 4 and new_button <= 10) or new_button == 13: #if system says to hold a button
#         if new_button == 5:
#             button("punch")
#         elif new_button == 6:
#             button("kick")
#         elif new_button == 7:
#             button("slash")
#         elif new_button == 8:
#             button("hslash")
#         elif new_button == 9:
#             button("dust")
#         elif new_button == 13:
#             button("dash")
#         elif new_button == 10:
#             button("punch")
#             button("kick")
#     if move == 0: #cases for movement
#         dpad(6)
#         sleep(frame*4)
#     elif move == 1:
#         dpad(4)
#         sleep(frame*4)
#     elif move == 2:
#         dpad(8)
#         sleep(frame*4)
#     elif move == 3:
#         dpad(2)
#         sleep(frame*4)
#     elif move == 4:
#         dpad(2)
#         sleep(frame)
#         dpad(3)
#         sleep(frame)
#         dpad(6)
#         sleep(frame*2)
#     elif move == 5:
#         dpad(2)
#         sleep(frame)
#         dpad(1)
#         sleep(frame)
#         dpad(4)
#         sleep(frame*2)
#     elif move == 6:
#         dpad(6)
#         sleep(frame)
#         dpad(2)
#         sleep(frame)
#         dpad(3)
#         sleep(frame*2)
#     elif move == 7:
#         dpad(4)
#         sleep(frame)
#         dpad(2)
#         sleep(frame)
#         dpad(3)
#         sleep(frame*2)
#     elif move == 8:
#         dpad(4)
#         sleep(frame)
#         dpad(1)
#         sleep(frame)
#         dpad(2)
#         sleep(frame)
#         dpad(6)
#         sleep(frame)
#     elif move == 9:
#         dpad(6)
#         sleep(frame)
#         dpad(2)
#         sleep(frame)
#         dpad(4)
#         sleep(frame)
#     elif move == 10:
#         dpad(1)
#         sleep(frame * 4)
#     elif move == 11:
#         dpad(3)
#         sleep(frame * 3)
#     elif move == 12:
#         dpad(7)
#         sleep(frame * 3)
#     elif move == 13:
#         dpad(9)
#         sleep(frame*4)
#     elif move == 14:
#         dpad(5)
#         sleep(frame * 4)
#
#     if new_button == 0: #cases for holding the button late
#         button("punch")
#     if new_button == 1:
#         button("kick")
#     if new_button == 2:
#         button("slash")
#     if new_button == 3:
#         button("hslash")
#     if new_button == 4:
#         button("dust")
#     if new_button ==10:
#         button("punch")
#         button("kick")
#     if new_button==11:
#         button("punch")
#         button("kick")
#         button("slash")
#     if new_button==12:
#         button("dust")
#         button("punch")
#     if new_button==13:
#         button("dash")
#     sleep(frame)
