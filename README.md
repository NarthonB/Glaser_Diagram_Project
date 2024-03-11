The purpose of this project is to visually and quantitatively show the points of condensation in a user-defined wall.

All data and equations were taken from ***********'s Applied Thermodynamics course at INSA Strasbourg (citation still to be approved).
Some early versions might have some language inconsistencies since this course was taught in french. 

Version 0.1: The most basic possible version of the code.
- Includes most of the relevant calculations required to get a temperature (and therefore pressure) evolution across the wall
- Returns all relevant evolutions on the same axis
- No user input - pre-designed wall


Version 1: Fully functional code with little to no extra functionalities.

  v1.0: Most simple version which still outputs the desired output.
  - allows user to customize a wall of (maximum 3) layers by their materials and thickness
  - returns a plot with a temperature and a pressure y-axis, which highlights where condensation occurs

  v1.1: More organised version.
  - use of arrays rather than variable names to shorten code and make it easier to add layers in the future 
  - clearly commented blocks of code
  - improved variable names

  v1.?: (list of planned features to be added during the v1 phase)
  - will integrate a choice of a permeable coating to mitigate condensation
  - will improve the visualisation with labels and/or textures for different materials
  - improve the LayerData file to make the types of material clearer
  - improve user interface


Version 2 (Future version): will build upon the latest v1 by adding extra functionalities
- use of the 'Meteostat_import_station_data' function included in the project files to be able to determine the validity of the model for different parts of the year for a given location in the world
- estimate how the materials will be affected by the condensation over time
- cost analysis
- try and integrate acoustics? 
