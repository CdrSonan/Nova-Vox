from kivy.properties import StringProperty, ObjectProperty, BooleanProperty, NumericProperty
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label

from kivy.clock import mainthread

from UI.code.editor.Util import ImageButton, ImageToggleButton
from UI.code.editor.Popups import SingerSettingsPanel

class SingerPanel(AnchorLayout):
    name = StringProperty()
    image = ObjectProperty()
    index = NumericProperty()
    def changeTrack(self):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.changeTrack(self.index)
    def openSettings(self):
        SingerSettingsPanel(self.index).open()
    def copyTrack(self):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.copyTrack(self.index, self.name, self.image)
    def deleteTrack(self):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.deleteTrack(self.index)
    def updateVolume(self, volume):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.updateVolume(self.index, volume)

class ParamPanel(ToggleButton):
    def __init__(self, name, switchable, sortable, deletable, index, switchState = True, **kwargs):
        super().__init__(**kwargs)
        self.name = StringProperty()
        self.switchable = BooleanProperty()
        self.sortable = BooleanProperty()
        self.index = NumericProperty()
        self.deletable = BooleanProperty()
        self.name = name
        self.switchable = switchable
        self.sortable = sortable
        self.deletable = deletable
        self.index = index
        self.background_color = (1, 1, 1, 0.3)
        self.makeWidgets(switchState)
    @mainthread
    def makeWidgets(self, switchState = True):
        self.add_widget(Label(size_hint = (None, None), size = (self.width - 106, 30), pos = (self.x + 103, self.y + 3), text = self.name))
        if self.sortable:
            self.add_widget(ImageButton(size_hint = (None, None), size = (40, 30), pos = (self.x + 33, self.y + 3), imageNormal = "UI/assets/ParamList/Adaptive03.png", imagePressed = "UI/assets/ParamList/Adaptive03_clicked.png", on_release = self.moveParam))
        if self.deletable:
            self.add_widget(ImageButton(size_hint = (None, None), size = (30, 30), pos = (self.x + 73, self.y + 3), imageNormal = "UI/assets/TrackList/SingerGrey03.png", imagePressed = "UI/assets/TrackList/SingerGrey03_clicked.png", on_press = self.deleteParam))
        if self.switchable:
            self.add_widget(ImageToggleButton(size_hint = (None, None), size = (30, 30), pos = (self.x + 3, self.y + 3), imageNormal = "UI/assets/ParamList/Adaptive02.png", imagePressed = "UI/assets/ParamList/Adaptive01.png", function = self.enableParam))
            if switchState:
                self.children[0].state = "down"
    def on_width(self, widget, width):
        for i in self.children:
            if i.__class__.__name__ == "Label":
                i.width = self.width - 106
                i.x = self.x + 103
    def enableParam(self):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        if self.children[0].state == "down":
            middleLayer.enableParam(self.index, self.name)
        else:
            middleLayer.disableParam(self.index, self.name)
    def moveParam(self):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        delta = 0
        middleLayer.moveParam(self.name, self.switchable, self.sortable, self.index, delta, self.children[0].state == "down")
    def deleteParam(self):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.deleteParam(self.index)
    def changeParam(self):
        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.changeParam(self.index, self.name)