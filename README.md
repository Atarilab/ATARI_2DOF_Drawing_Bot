# MIRMI_2DOF_Drawing_Bot
 
This respository contains all the CAD files, firmware and API to build an use the 2 DOF drawing robot by MIRMI lab.

# API
Use the API to interface with the drawing robot system:

## Includes
First of, import the api and the predefined shapes via:
```
from drawing_bot_api import Drawing_bot
from drawing_bot_api import shapes
```

## Intialisation
To communication with the robot you need to create an instance of Drawing_bot class:
```
drawing_bot = Drawing_bot()
```
Parameters:
- unit: Changes the unit used to describe the robots position within the working area (default: 'mm') \
  options:
  * millimeter: 'mm'
  * centimeter: 'cm'
  * meter: 'm'
- speed: Changs the speed at which the drawing robot runs in unit/sec (default: 200)
