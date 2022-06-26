#define nvxAssocName "Nova-Vox Project File"
#define nvxAssocExt ".nvx"
#define nvxAssocKey "NovaVox" + nvxAssocExt
#define nvvbAssocName "Nova-Vox Voicebank"
#define nvvbAssocExt ".nvvb"
#define nvvbAssocKey "NovaVox" + nvvbAssocExt
#define nvprAssocName "Nova-Vox universal parameter"
#define nvprAssocExt ".nvpr"
#define nvprAssocKey "NovaVox" + nvprAssocExt

#define nvvbAssocKeyDK "NovaVoxDK" + nvvbAssocExt
#define nvprAssocKeyDK "NovaVoxDK" + nvprAssocExt

[Setup]
; NOTE: The value of AppId uniquely identifies this application. Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{680531A2-5B3C-47B5-8380-CA6D7033BF13}
AppName="Nova-Vox"
AppVersion="Alpha 3.0"
AppVerName="Nova-Vox Alpha 3.0"
AppPublisher="Nova-Vox development team"
AppPublisherURL="https://nova-vox.org/"
AppSupportURL="https://nova-vox.org/"
AppUpdatesURL="https://nova-vox.org/"
AppComments="Nova-Vox hybrid vocal synthesizer"
AppContact="https://nova-vox.org/"
AppReadmeFile="https://nova-vox.org/tutorial/"
DefaultDirName={autopf}\Nova-Vox
DefaultGroupName="Nova-Vox"
ChangesAssociations=yes
LicenseFile=D:\Nova-Vox\GitHub\Nova-Vox\license.txt
InfoBeforeFile=D:\Nova-Vox\GitHub\Nova-Vox\info_pre_install.txt
InfoAfterFile=D:\Nova-Vox\GitHub\Nova-Vox\info_post_install.txt
OutputDir="D:\Nova-Vox\GitHub\Nova-Vox\"
OutputBaseFilename=Nova-Vox setup
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern

[Types]
Name: "full"; Description: "Full installation"
Name: "minimal"; Description: "Minimal installation"
Name: "custom"; Description: "Custom installation"; Flags: iscustom

[Components]
Name: "main"; Description: "Nova-Vox Editor and Dependencies"; Types: full minimal custom; Flags: fixed
Name: "devkit"; Description: "Devkit Executable"; Types: full custom
Name: "voices"; Description: "Default Voicebanks"; Types: full custom
Name: "voices\Adachi_Rei"; Description: "Adachi Rei UTAU port"; Types: full custom
Name: "params"; Description: "Default Parameters"; Types: full custom

[Tasks]
Name: "desktopiconeditor"; Description: "{cm:CreateDesktopIcon} (Editor)"; GroupDescription: "{cm:AdditionalIcons}"
Name: "desktopicondevkit"; Description: "{cm:CreateDesktopIcon} (Devkit)"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Dirs]
Name: {code:GetDataDir}; Flags: uninsneveruninstall
Name: "{code:GetDataDir}\Voices"; Flags: uninsneveruninstall
Name: "{code:GetDataDir}\Parameters"; Flags: uninsneveruninstall
Name: "{code:GetDataDir}\Addons"; Flags: uninsneveruninstall
Name: "{userappdata}\Nova-Vox\Logs"


[Code]
var
  DataDirPage: TInputDirWizardPage;
procedure InitializeWizard;
begin
  DataDirPage := CreateInputDirPage(wpSelectDir,
    'Select Data Directory', 'Where should Voicebanks, Parameters and Addon files be stored?',
    'Select the folder in which Setup should install Voicebanks, Parameters and Addon files, then click Next. You can change this folder later in the settings panel.',
    False, '');
  DataDirPage.Add('');
  DataDirPage.Values[0] := ExpandConstant('{userappdata}\Nova-Vox');
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo,
  MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
var
  S: String;
begin
  S := '';
  S := S + MemoDirInfo + NewLine;
  S := S + Space + DataDirPage.Values[0] + ' (data files)' + NewLine;
  S := S + MemoTypeInfo + NewLine;
  if WizardSetupType(False) = 'custom' then
    S := S + MemoComponentsInfo + NewLine;
  S := S + MemoGroupInfo + NewLine;
  S := S + MemoTasksInfo + NewLine;
  Result := S;
end;

function GetDataDir(Param: String): String;
begin
  Result := DataDirPage.Values[0];
end;

[Files]
Source: "{SourcePath}\dist\Nova-Vox\*"; DestDir: "{app}"; Components: main; Excludes: "Nova-Vox Devkit.exe"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{SourcePath}\settings.ini"; DestDir: "{userappdata}\Nova-Vox"; Components: main; Flags: ignoreversion
Source: "{SourcePath}\dist\Nova-Vox\Nova-Vox Devkit.exe"; DestDir: "{app}"; Components: devkit; Flags: ignoreversion
Source: "{SourcePath}\Voices\Adachi Rei.nvvb"; DestDir: "{code:GetDataDir}\Voices"; Components: voices\Adachi_Rei; Flags: ignoreversion
;Source: "{SourcePath}\Params\*"; DestDir: "{code:GetDataDir}\Parameters"; Components: params; Flags: ignoreversion
;Source: "{SourcePath}\Addons\*"; DestDir: "{code:GetDataDir}\Addons"; Flags: ignoreversion recursesubdirs createallsubdirs

[INI]
Filename: "{userappdata}\Nova-Vox\settings.ini"; Section: "Dirs"
Filename: "{userappdata}\Nova-Vox\settings.ini"; Section: "Dirs"; Key: "dataDir"; String: "{code:GetDataDir}"

[Icons]
Name: "{group}\Nova-Vox\Nova-Vox Editor"; Filename: "{app}\Nova-Vox Editor.exe"
Name: "{autodesktop}\Nova-Vox Editor"; Filename: "{app}\Nova-Vox Editor.exe"; Tasks: desktopiconeditor
Name: "{group}\Nova-Vox\Nova-Vox Devkit"; Filename: "{app}\Nova-Vox Devkit.exe"
Name: "{autodesktop}\Nova-Vox Devkit"; Filename: "{app}\Nova-Vox Devkit.exe"; Tasks: desktopicondevkit

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Registry]
Root: HKA; Subkey: "Software\Classes\{#nvxAssocExt}\OpenWithProgids"; ValueType: string; ValueName: "{#nvxAssocKey}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\{#nvxAssocKey}"; ValueType: string; ValueName: ""; ValueData: "{#nvxAssocName}"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\{#nvxAssocKey}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\Nova-Vox Editor.exe,0"
Root: HKA; Subkey: "Software\Classes\{#nvxAssocKey}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\Nova-Vox Editor.exe"" ""%1"""
Root: HKA; Subkey: "Software\Classes\Applications\Nova-Vox Editor.exe\SupportedTypes"; ValueType: string; ValueName: {#nvxAssocExt}; ValueData: ""

Root: HKA; Subkey: "Software\Classes\{#nvvbAssocExt}\OpenWithProgids"; ValueType: string; ValueName: "{#nvvbAssocKey}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\{#nvvbAssocKey}"; ValueType: string; ValueName: ""; ValueData: "{#nvvbAssocName}"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\{#nvvbAssocKey}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\Nova-Vox Editor.exe,0"
Root: HKA; Subkey: "Software\Classes\{#nvvbAssocKey}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\Nova-Vox Editor.exe"" ""%1"""
Root: HKA; Subkey: "Software\Classes\Applications\Nova-Vox Editor.exe\SupportedTypes"; ValueType: string; ValueName: {#nvvbAssocExt}; ValueData: ""
Root: HKA; Subkey: "Software\Classes\Applications\Nova-Vox Devkit.exe\SupportedTypes"; ValueType: string; ValueName: {#nvvbAssocExt}; ValueData: ""; Components: devkit

Root: HKA; Subkey: "Software\Classes\{#nvprAssocExt}\OpenWithProgids"; ValueType: string; ValueName: "{#nvprAssocKey}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\{#nvprAssocKey}"; ValueType: string; ValueName: ""; ValueData: "{#nvprAssocName}"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\{#nvprAssocKey}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\Nova-Vox Editor.exe,0"
Root: HKA; Subkey: "Software\Classes\{#nvprAssocKey}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\Nova-Vox Editor.exe"" ""%1"""
Root: HKA; Subkey: "Software\Classes\Applications\Nova-Vox Editor.exe\SupportedTypes"; ValueType: string; ValueName: {#nvprAssocExt}; ValueData: ""
Root: HKA; Subkey: "Software\Classes\Applications\Nova-Vox Devkit.exe\SupportedTypes"; ValueType: string; ValueName: {#nvprAssocExt}; ValueData: ""; Components: devkit


Root: HKA; Subkey: "Software\Classes\{#nvvbAssocExt}\OpenWithProgids"; ValueType: string; ValueName: "{#nvvbAssocKeyDK}"; ValueData: ""; Flags: uninsdeletevalue; Components: devkit
Root: HKA; Subkey: "Software\Classes\{#nvvbAssocKeyDK}"; ValueType: string; ValueName: ""; ValueData: "{#nvvbAssocName}"; Flags: uninsdeletekey; Components: devkit
Root: HKA; Subkey: "Software\Classes\{#nvvbAssocKeyDK}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\Nova-Vox Editor.exe,0"; Components: devkit
Root: HKA; Subkey: "Software\Classes\{#nvvbAssocKeyDK}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\Nova-Vox Devkit.exe"" ""%1"""; Components: devkit

Root: HKA; Subkey: "Software\Classes\{#nvprAssocExt}\OpenWithProgids"; ValueType: string; ValueName: "{#nvprAssocKeyDK}"; ValueData: ""; Flags: uninsdeletevalue; Components: devkit
Root: HKA; Subkey: "Software\Classes\{#nvprAssocKeyDK}"; ValueType: string; ValueName: ""; ValueData: "{#nvprAssocName}"; Flags: uninsdeletekey; Components: devkit
Root: HKA; Subkey: "Software\Classes\{#nvprAssocKeyDK}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\Nova-Vox Editor.exe,0"; Components: devkit
Root: HKA; Subkey: "Software\Classes\{#nvprAssocKeyDK}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\Nova-Vox Devkit.exe"" ""%1"""; Components: devkit

[Run]
Filename: "{app}\Nova-Vox Editor.exe"; Description: "{cm:LaunchProgram,{#StringChange("Nova-Vox", '&', '&&')}}"; Flags: nowait postinstall skipifsilent

