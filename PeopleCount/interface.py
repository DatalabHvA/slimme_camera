"""
interface.py

PURPOSE
-------
This script acts as a very early proof-of-concept interface between:
- The vision modules (Vision_A.py / Vision_B.py)
- A future higher-level control or UI system

Currently, it:
- Runs ONE vision script via a subprocess
- Reads its output
- Prints the detected number of people

IMPORTANT LIMITATIONS
---------------------
⚠ This file is NOT finished.
⚠ This file is NOT production-ready.

Known missing features:
- No parallel execution of Vision_A and Vision_B
- No parsing of camera ID or wall state
- No error recovery or retry logic
- No synchronization between rooms
- No message passing or IPC beyond subprocess stdout

This file mainly demonstrates:
✔ How a vision module can be called externally
✔ How results can be returned via stdout
"""

# ------------------ IMPORTS ------------------ #
import subprocess   #Used to run external Python scripts


# ------------------ VISION INTERFACE ------------------ #
def get_people_count():
    """
    Runs a vision module (Vision_A.py OR Vision_B.py)
    as a subprocess and retrieves the detected people count.

    Current behavior:
    - Executes only ONE vision script
    - Assumes the script prints ONLY a number
    - Returns that number as an integer

    NOTE:
    Vision_A.py / Vision_B.py are expected to be called like this:
        python3 vision_A.py

    Future versions should:
    - Run both cameras
    - Parse CSV output (CAMERA_ID,WALL_STATE,COUNT)
    - Merge results intelligently

    :return: number of people detected
    :rtype: int
    """

    try:
        #Run the vision script as a subprocess
        result = subprocess.check_output(
            [
                "python3",
                "vision_A.py"      #Change to vision_B.py for the second camera
            ],
            text=True,             #Capture stdout as a string
            timeout=120            #Safety timeout (seconds)
        )

        #Remove whitespace and convert output to integer
        return int(result.strip())

    except subprocess.TimeoutExpired:
        #Vision script took too long (camera freeze, model stall, etc.)
        print("Vision process timed out")
        return -1

    except ValueError:
        #Output was not a valid integer
        #(e.g. unexpected debug print statements)
        print("Invalid response from vision module")
        return -1


# ------------------ ENTRY POINT ------------------ #
if __name__ == "__main__":
    """
    Standalone test execution.

    This allows interface.py to be run directly
    to verify that the vision subprocess call works.
    """

    people = get_people_count()
    print(f"People in room: {people}")