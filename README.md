The purpose of this project is to visually and quantitatively show the area of condensation in a user-defined wall. These plots are not accurate representations of condensation in the wall, only indicating the risk of the phenomenon. 

All data and equations were taken from the Applied Thermodynamics course taken at INSA Strasbourg (cited below).
Some early versions might have some language inconsistencies since this course was taught in french. 

Walther, E. (2024). _Thermodynamique Appliquée; Air Humide et Transfert de Masse_ [Class Handout]. Retrieved from https://moodle.insa-strasbourg.fr/course/view.php?id=1251

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Version 0.1: The most basic possible version of the code.
- Includes most of the relevant calculations required to get a temperature (and therefore pressure) evolution across the wall
- Returns all relevant evolutions on the same axis
- No user input - pre-designed wall


Version 1: Fully functional code with little to no extra functionalities.

  v1.0: Most simple version which still returns the desired output.
  - allows user to customize a wall of (maximum 3) layers by their materials and thickness
  - returns a plot with a temperature and a pressure y-axis, which highlights where condensation occurs

  v1.1: More organised version.
  - use of arrays rather than variable names to shorten code and make it easier to add layers in the future 
  - clearly commented blocks of code
  - improved variable names

  v1.2: Layer Properties and Internal/External Conditions assigned to DataFrames.
  - Clearer to differentiate constant and manipulated variables.
  - CONCERN: While more organised, lines of code are much more long-winded. Could make code visually harder to understand.

  v1.3: 
  - reorganised code, some variable name changes and improved commenting
  - defined equations separately from the main code
  - changed layers dataframe
  - added a function to find condensation point(s)
  - added convection coefficient equation depending on terms of air speed

v1.4:
  - condensed all final data into a dataframe
  - changed plotting function to take df and layer bounds as input (hopefully make this part of df in future)
  - Add vapour barrier
    - changed vapour flow equation to include addition of a VB
  - changed equation for checking layer (not sure why/how it worked before)
  - reduced number of variables
  - created a separate rounding function
  - Improved plot display
    - created a colour-picking function to clarify different materials
    - made separations clearer
  - Reorganised some code

All code after this point was linked to git and versions were no longer uploaded as files.


Version 2 : Final Version
- Full dash app layout
- Use of weather station data (Meteostat) to test the wall at different points of the year for a given location in the world
- fully debugged (some problems with US stations as well as stations with limited data)
- Internal conditions are now modifiable 
- Maximum of 3 layers (dash editable datatable with a dropdown was near impossible to make dynamic)
