from kivy.properties import StringProperty, ObjectProperty, BooleanProperty, NumericProperty
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label

from kivy.clock import mainthread

from UI.code.editor.Util import ImageButton, ImageToggleButton
from UI.code.editor.Popups import SingerSettingsPanel

class SingerPanel(AnchorLayout):
    """Header widget for a vocal track"""

    name = StringProperty()
    image = ObjectProperty()
    index = NumericProperty()

    def changeTrack(self) -> None:
        """signals a change of the active track to the middleLayer"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.changeTrack(self.index)

    def openSettings(self) -> None:
        """opens the settings panel of the track"""

        SingerSettingsPanel(self.index).open()

    def copyTrack(self) -> None:
        """signals the middleLayer to duplicate the track"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.copyTrack(self.index, self.name, self.image)

    def deleteTrack(self) -> None:
        """signals the middleLayer to delete the track"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.deleteTrack(self.index)

    def updateVolume(self, volume:float) -> None:
        """signals the middleLayer to update the volume of the track"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.updateVolume(self.index, volume)

class ParamPanel(ToggleButton):
    """Header widget for a resampler parameter, or tuning curve used for audio processing"""

    def __init__(self, name:str, switchable:bool, sortable:bool, deletable:bool, index:int, switchState:bool = True, **kwargs) -> None:
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
    def makeWidgets(self, switchState:bool = True) -> None:
        """Initialises all sub-widgets of the header widget"""

        self.add_widget(Label(size_hint = (None, None), size = (self.width - 106, 30), pos = (self.x + 103, self.y + 3), text = self.name))
        if self.sortable:
            self.add_widget(ImageButton(size_hint = (None, None), size = (40, 30), pos = (self.x + 33, self.y + 3), imageNormal = "UI/assets/ParamList/Adaptive03.png", imagePressed = "UI/assets/ParamList/Adaptive03_clicked.png", on_release = self.moveParam))
        if self.deletable:
            self.add_widget(ImageButton(size_hint = (None, None), size = (30, 30), pos = (self.x + 73, self.y + 3), imageNormal = "UI/assets/TrackList/SingerGrey03.png", imagePressed = "UI/assets/TrackList/SingerGrey03_clicked.png", on_press = self.deleteParam))
        if self.switchable:
            self.add_widget(ImageToggleButton(size_hint = (None, None), size = (30, 30), pos = (self.x + 3, self.y + 3), imageNormal = "UI/assets/ParamList/Adaptive02.png", imagePressed = "UI/assets/ParamList/Adaptive01.png", function = self.enableParam))
            if switchState:
                self.children[0].state = "down"

    def on_width(self, widget, width) -> None:
        """updates the width of the label subwidget when the width of the widget changes"""
        for i in self.children:
            if i.__class__.__name__ == "Label":
                i.width = self.width - 106
                i.x = self.x + 103

    def enableParam(self) -> None:
        """signals the middleLayer to enable or disable the parameter associated with the widget when its toggle is switched"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        if self.children[0].state == "down":
            middleLayer.enableParam(self.index, self.name)
        else:
            middleLayer.disableParam(self.index, self.name)

    def moveParam(self) -> None:
        """signals the middleLayer to move the widget to a new position within the stack"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        delta = 0
        middleLayer.moveParam(self.name, self.switchable, self.sortable, self.index, delta, self.children[0].state == "down")

    def deleteParam(self) -> None:
        """signals the middleLayer to delete the widget and associated tuning curve"""

        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.deleteParam(self.index)

    def changeParam(self) -> None:
        """signals the middleLayer a change of the active parameter, prompting the required UI updates"""
        
        global middleLayer
        from UI.code.editor.Main import middleLayer
        middleLayer.changeParam(self.index, self.name)
