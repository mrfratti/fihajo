# Jenkins Environment Variable Setup Guide

Follow this guide in case of any issues with system binaries like `sh`, `pythin` and `pip` commands in Jenkins.
These steps ensures that your build process can locate and use necessary executables without issues across Mac,
Windows and Linux environments. 

## Step 1: Verify Executable Paths

Before setting up your Jenkins environment, you first need to confirm the paths of the executables that are needed
on your system. 

**Mac and Linux**

Open your terminal and type the following commands to find the paths:

```bash
which sh
```
Output: `/bin/sh`

```bash
which python3
```
Output: `usr/bin/python3`

```bash
which pip3
```
Output: `usr/local/bin/pip3`


Output of these commands returns the path to the each of executables if they are installed.

**Windows**

If you use Windows operating system you should open command prompt and type the following commands:

```cmd
where sh
```

```cmd
where python
```

```cmd
where pip
```

The output should look something like:

- `C:\ProgramData\chocolatey\bin\sh.exe`
- `C:\Users\username\AppData\Local\Programs\Python\Python39\python.exe`
- `C:\Users\username\AppData\Local\Programs\Python\Python39\Scripts\pip.exe`


## Step 2: Setting Up Environment Variables in Jenkins

To set or modify envinronment variables in Jenkins you have to:

1. Navigate to **Manage Jenkins > System Configration > System**
2. Scroll down to **Global properties**
3. Check the **Environment variables** box
4. Add new variables by clicking **Add**
5. Apply the changes by cliking **Save**

**Example Configuration**

- Name: PATH
- Value for Mac/Linux:
  - `/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin:$PATH`
- Value for Windows:
  - `C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem;YOUR_PYTHON_PATH;YOUR_PIP_PATH`

