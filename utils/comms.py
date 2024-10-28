import re
import roman
import mariadb


def connect_to_DB(username,
                  password,
                  server = "laser365-1.chem.msu.ru",
                  port=3306,
                  database = "kurucz"):
    
    conn = mariadb.connect(
            user=username,
            password=password,
            host=server,
            port=port,
            database=database)
    
    return conn
    
    
def is_valid_element(symbol):
    # List of all valid element symbols in the periodic table
    valid_elements = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    ]
    return symbol.capitalize() in valid_elements
    
    
def convert_species_request(s):
    def parse_roman_part(part):
        part = part.strip()
        if not part:
            return []
        if '-' in part:
            start, end = [p.strip() for p in part.split('-')]
            return list(range(roman.fromRoman(start.upper())-1, roman.fromRoman(end.upper()) + 1))
        return [roman.fromRoman(part.upper())-1]

    def parse_ionization_stages(stages):
        # Separate by commas, then handle each part
        parts = re.split(r'\s*,\s*', stages)
        ionization = []
        for part in parts:
            ionization.extend(parse_roman_part(part))
        return ionization

    elements = []
    ionizations = []
    chem_elems = s.split(';')
    
    for chem_elem in chem_elems:
        match = re.match(r'([A-Za-z]{1,2})\s*([ivx,\s-]*)', chem_elem.strip(), re.IGNORECASE)
        if match:
            element = match.group(1).capitalize()
            if is_valid_element(element) == False:
                raise ValueError(f"Chemical element symbol {element} is incorrect")
            if match.group(2).strip():
                ionization = parse_ionization_stages(match.group(2))
            else:
                ionization = 'All'
            elements.append(element)
            ionizations.append(ionization)
        else:
            raise ValueError("Invalid input format")
    
    return elements, ionizations