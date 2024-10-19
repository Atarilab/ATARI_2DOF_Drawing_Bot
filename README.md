# MIRMI_2DOF_Drawing_Bot
 
This respository contains all the firmware and API to program and interface with the 2 DOF drawing robot by MIRMI lab.

# API
Use the API to interface with the drawing robot.

## Includes
First of, import the api and the predefined shapes via:
```
from drawing_bot_api import Drawing_bot
from drawing_bot_api import shapes
```

## Intialisation
To communication with the robot you need to create an instance of the `Drawing_bot` class:
```
drawing_bot = Drawing_bot() # Creates an instance with default parameters
```
Parameters:
- unit: Changes the unit used to describe the robots position within the working area (default: 'mm') \
  options:
  * millimeter: 'mm'
  * centimeter: 'cm'
  * meter: 'm'
- speed: Changes the speed at which the drawing robot runs in unit/sec (default: 200)

```
# example
drawing_bot = Drawing_bot(unit='m', speed=400)
```

## Add shapes
To create a program sequence you add shapes to your instance of the `Drawing_bot` class.
```
# example (in millimeters)
drawing_bot.add_shape(shapes.Line([20, 80], [40, 120]))
```

### Point
TODO

### Line
To draw a line use `shapes.Line(start_point, end_point)`
Parameters:
- start_point: Defines where the line should start
- end_point: Defines where the line should end
```
# example (in millimeters)
shapes.Line([20, 80], [40, 120])
```

### Circle
To draw a circle use `shapes.Circle(center_point, radius)`
Parameters:
- center_point: Defines the center of the circle
- radius: Defines the radius of the circle
```
# example (in millimeters)
shapes.Circle([0, 100], 20)
```

### Partial Circle
To draw a circle that's not fully completed use `shapes.Partial_circle(start_point, end_point, radius, direction)`
Parameters:
- start_point: Defines the start point of the partial circle
- end_point: Defines the end point of the partial circle
- radius: Defines the radius of the partial circle
- direction: Defines wether the partial circle should be drawn clockwise (`direction=1`) or anti-clockwise (`direction=-1`)
optional:
- big_angle: Defines whether the smaller (`big_angle=False`) or bigger (`big_angle=True`) part of the partial circle is drawn. Default: `False`
```
# example (in millimeters)
shapes.Partial_circle([-30, 100], [30, 100], 40, 1, big_angle=True)
```

## Plot drawing
To see what your program sequence looks like before sending it to the drawing robot you can create a plot of the drawing.
```
drawing_bot.plot()
```
Optional parameters:
- resolution: Defines how many points per unit are drawn to create the plot (default: 2)
- blocking: Defines wether program flow stops (`blocking=True`) or continues (`blocking=False`); default: `True`

## Send commands to robot
To send the whole program sequence as commands to the robot use
```
drawing_bot.execute()
```
Optional parameters:
- promting: If set to `True` the user will get a promt to confirm the execution of the program before it is send to the robot. Default: `True`
- clear_buffer: If set to `True` the program sequence will be erased from the buffer of the instance of the Drawing_bot class. Default: `True`

## Force reset
In case the execution of the program code is causing problems because the serial connection to the robot isn't working, you can use `drawing_bot.hard_reset()` to reset the whole system and re-initialise the serial connection.\
You can call this function before every program execution but forcing a reset will lead to the robot redoing it's zeroing and initialisation routine which can be annoying. So it is recommended to only use this function if problems arrise.





