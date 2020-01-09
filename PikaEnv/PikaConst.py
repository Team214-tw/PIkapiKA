from pynput.keyboard import Key

WIDTH = 480
HEIGHT = 360
ACTIONS = [
    (Key.up,),
    (Key.left,),
    (Key.right,),
    (Key.enter,),
    (Key.up, Key.left),
    (Key.up, Key.right),
    (Key.up, Key.enter),
    (Key.left, Key.enter),
    (Key.right, Key.enter),
    (Key.down, Key.enter),
    (),
]