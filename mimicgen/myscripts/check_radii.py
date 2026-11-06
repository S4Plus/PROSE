#检查冲突半径：0.1326274169979695
import numpy as np
from robosuite.models.objects import SquareNutObject, RoundNutObject
from robosuite.models.arenas import PegsArena
from robosuite.utils.mjcf_utils import string_to_array

def check_radii():
    # Check Nut Radii
    square_nut = SquareNutObject(name="square_nut")
    round_nut = RoundNutObject(name="round_nut")
    
    print(f"SquareNut horizontal_radius: {square_nut.horizontal_radius}")
    print(f"RoundNut horizontal_radius: {round_nut.horizontal_radius}")
    
    # Check Peg Radius
    arena = PegsArena()
    peg1_xml = arena.worldbody.find("./body[@name='peg1']")
    peg1_size = string_to_array(peg1_xml.find("./geom").get("size"))
    peg1_radius = np.linalg.norm(peg1_size[0:2], 2)
    print(f"Peg1 calculated horizontal_radius: {peg1_radius}")
    
    print(f"Sum (Square + Peg): {square_nut.horizontal_radius + peg1_radius}")
    print(f"Sum (Round + Peg): {round_nut.horizontal_radius + peg1_radius}")

if __name__ == "__main__":
    check_radii()
