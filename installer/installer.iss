;Copyright 2022 - 2024 Contributors to the Nova-Vox project

;This file is part of Nova-Vox.
;Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
;Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
;You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

#define nvxAssocName "Nova-Vox Project File"
#define nvxAssocExt ".nvx"
#define nvxAssocKey "NovaVox" + nvxAssocExt
#define nvvbAssocName "Nova-Vox Voicebank"
#define nvvbAssocExt ".nvvb"
#define nvvbAssocKey "NovaVox" + nvvbAssocExt
;#define nvprAssocName "Nova-Vox universal parameter"
;#define nvprAssocExt ".nvpr"
;#define nvprAssocKey "NovaVox" + nvprAssocExt

#define nvvbAssocKeyDK "NovaVoxDK" + nvvbAssocExt
;#define nvprAssocKeyDK "NovaVoxDK" + nvprAssocExt

[Setup]
; NOTE: The value of AppId uniquely identifies this application. Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{680531A2-5B3C-47B5-8380-CA6D7033BF13}
AppName="Nova-Vox"
AppVersion="0.99.0"
AppVerName="Nova-Vox 0.99.0"
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
LicenseFile=..\COPYING.txt
InfoBeforeFile=.\info_pre_install.txt
InfoAfterFile=.\info_post_install.txt
OutputDir="..\"
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
Name: "devkit\phontables"; Description: "Phonetic tables and presets for Devkit"; Types: full custom
Name: "devkit\aiv1"; Description: "V1 Japanese+English AI base model"; Types: full custom
Name: "voices"; Description: "Default Voicebanks"; Types: full custom
;Name: "voices\TYC"; Description: "Tsukuyomi-chan demo Voicebank"; Types: full custom
;Name: "voices\Arachne"; Description: "Arachne Japanese UTAU port"; Types: full custom
;Name: "params"; Description: "Default Parameters"; Types: full custom
;Name: "voices\Barrels"; Description: "Barrels test Voicebank"; Types: full custom

[Tasks]
Name: "desktopiconeditor"; Description: "{cm:CreateDesktopIcon} (Editor)"; GroupDescription: "{cm:AdditionalIcons}"
Name: "desktopicondevkit"; Description: "{cm:CreateDesktopIcon} (Devkit)"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Dirs]
Name: {code:GetDataDir}; Flags: uninsneveruninstall
Name: "{code:GetDataDir}\Voices"; Flags: uninsneveruninstall
Name: "{code:GetDataDir}\Parameters"; Flags: uninsneveruninstall
Name: "{code:GetDataDir}\Addons"; Flags: uninsneveruninstall
Name: "{code:GetDataDir}\Devkit_Phonetics\IPAConversions"; Components: devkit
Name: "{code:GetDataDir}\Devkit_Phonetics\Lists"; Components: devkit
Name: "{code:GetDataDir}\Devkit_Phonetics\UtauConversions"; Components: devkit
Name: "{code:GetDataDir}\Devkit_Presets\Dictionaries"; Components: devkit
;Name: "{code:GetDataDir}\Devkit_Presets\TrAis"; Components: devkit
Name: "{code:GetDataDir}\Devkit_Presets\MainAis"; Components: devkit
Name: "{userappdata}\Nova-Vox\Logs"


[Code]
var
  DataDirPage: TInputDirWizardPage;
var
  DownloadPage: TDownloadWizardPage;

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  if Progress = ProgressMax then
    Log(Format('Successfully downloaded file to {tmp}: %s', [FileName]));
  Result := True;
end;

procedure InitializeWizard;
begin
  DataDirPage := CreateInputDirPage(wpSelectDir,
    'Select Data Directory', 'Where should Voicebanks and other files be stored?',
    'Select the folder in which Setup should install Voicebanks, Presets, phoneme tables, AI base models and Addon files, then click Next. You can change this folder later in the settings panel.',
    False, '');
  DataDirPage.Add('');
  DataDirPage.Values[0] := ExpandConstant('{commondocs}\Nova-Vox');
  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing), SetupMessage(msgPreparingDesc), @OnDownloadProgress);
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

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  if CurPageID = wpReady then begin
    DownloadPage.Clear;
    // Use AddEx to specify a username and password
    if WizardIsComponentSelected('devkit\aiv1') then
      DownloadPage.Add('https://dl.nova-vox.org/jp-en-v1.hdf5', 'jp-en-v1.hdf5', '');
    DownloadPage.Show;
    try
      try
        DownloadPage.Download; // This downloads the files to {tmp}
        Result := True;
      except
        if DownloadPage.AbortedByUser then
          Log('Aborted by user.')
        else
          SuppressibleMsgBox(AddPeriod(GetExceptionMessage), mbCriticalError, MB_OK, IDOK);
        Result := False;
      end;
    finally
      DownloadPage.Hide;
    end;
  end else
    Result := True;
end;


[Files]
Source: "..\dist\Nova-Vox\*"; DestDir: "{app}"; Components: main; Excludes: "Nova-Vox Devkit.exe"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\assets\settings.ini"; DestDir: "{userappdata}\Nova-Vox"; Components: main; Flags: ignoreversion
Source: "..\dist\Nova-Vox\Nova-Vox Devkit.exe"; DestDir: "{app}"; Components: devkit; Flags: ignoreversion
Source: "{tmp}\jp-en-v1.hdf5"; DestDir: "{code:GetDataDir}\Devkit_Presets\MainAIs"; Components: devkit\aiv1; Flags: ignoreversion uninsneveruninstall external
;Source: "Params\*"; DestDir: "{code:GetDataDir}\Parameters"; Components: params; Flags: ignoreversion uninsneveruninstall
;Source: "Addons\*"; DestDir: "{code:GetDataDir}\Addons"; Flags: ignoreversion recursesubdirs createallsubdirs uninsneveruninstall
Source: "..\assets\Devkit_Phonetics\*"; DestDir: "{code:GetDataDir}\Devkit_Phonetics"; Components: devkit\phontables; Flags: ignoreversion recursesubdirs createallsubdirs uninsneveruninstall
Source: "..\assets\templates\dicts*"; DestDir: "{code:GetDataDir}\Devkit_Presets\Dictionaries"; Components: devkit\phontables; Flags: ignoreversion recursesubdirs createallsubdirs uninsneveruninstall 


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

;Root: HKA; Subkey: "Software\Classes\{#nvprAssocExt}\OpenWithProgids"; ValueType: string; ValueName: "{#nvprAssocKey}"; ValueData: ""; Flags: uninsdeletevalue
;Root: HKA; Subkey: "Software\Classes\{#nvprAssocKey}"; ValueType: string; ValueName: ""; ValueData: "{#nvprAssocName}"; Flags: uninsdeletekey
;Root: HKA; Subkey: "Software\Classes\{#nvprAssocKey}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\Nova-Vox Editor.exe,0"
;Root: HKA; Subkey: "Software\Classes\{#nvprAssocKey}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\Nova-Vox Editor.exe"" ""%1"""
;Root: HKA; Subkey: "Software\Classes\Applications\Nova-Vox Editor.exe\SupportedTypes"; ValueType: string; ValueName: {#nvprAssocExt}; ValueData: ""
;Root: HKA; Subkey: "Software\Classes\Applications\Nova-Vox Devkit.exe\SupportedTypes"; ValueType: string; ValueName: {#nvprAssocExt}; ValueData: ""; Components: devkit


Root: HKA; Subkey: "Software\Classes\{#nvvbAssocExt}\OpenWithProgids"; ValueType: string; ValueName: "{#nvvbAssocKeyDK}"; ValueData: ""; Flags: uninsdeletevalue; Components: devkit
Root: HKA; Subkey: "Software\Classes\{#nvvbAssocKeyDK}"; ValueType: string; ValueName: ""; ValueData: "{#nvvbAssocName}"; Flags: uninsdeletekey; Components: devkit
Root: HKA; Subkey: "Software\Classes\{#nvvbAssocKeyDK}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\Nova-Vox Editor.exe,0"; Components: devkit
Root: HKA; Subkey: "Software\Classes\{#nvvbAssocKeyDK}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\Nova-Vox Devkit.exe"" ""%1"""; Components: devkit

;Root: HKA; Subkey: "Software\Classes\{#nvprAssocExt}\OpenWithProgids"; ValueType: string; ValueName: "{#nvprAssocKeyDK}"; ValueData: ""; Flags: uninsdeletevalue; Components: devkit
;Root: HKA; Subkey: "Software\Classes\{#nvprAssocKeyDK}"; ValueType: string; ValueName: ""; ValueData: "{#nvprAssocName}"; Flags: uninsdeletekey; Components: devkit
;Root: HKA; Subkey: "Software\Classes\{#nvprAssocKeyDK}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\Nova-Vox Editor.exe,0"; Components: devkit
;Root: HKA; Subkey: "Software\Classes\{#nvprAssocKeyDK}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\Nova-Vox Devkit.exe"" ""%1"""; Components: devkit


[Run]
Filename: "{app}\Nova-Vox Editor.exe"; Description: "{cm:LaunchProgram,{#StringChange("Nova-Vox", '&', '&&')}}"; Flags: nowait postinstall skipifsilent


[UninstallDelete]
Type: filesandordirs; Name: "{userappdata}\Nova-Vox\Logs"
Type: dirifempty; Name: "{userappdata}\Nova-Vox"
