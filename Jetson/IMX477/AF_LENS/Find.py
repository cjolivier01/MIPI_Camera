# from arducam_config_parser import config_file_parser
import arducam

for bus_num in range(17):
  try:
      camera = arducam.mipi_camera()
      camera.init_camera(bus_num=bus_num)  # Try bus bus_num
      print(f"Initialized camera on bus {bus_num}!")
      break
  except:
      print(f"Camera not found on bus {bus_num}")
