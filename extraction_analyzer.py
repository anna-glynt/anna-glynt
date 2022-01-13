from pathlib import Path
from MapAnalyzer import MapAnalyzer


'''
This script orchestrates the diagnostic flow. It asks for the user to input
an extraction uuid and a field; from there, it calls the necessary functions
and classes and displays information.
'''

extract_uuid = input('Extraction uuid: ')
field = input('Field: ')
extract_target = input('Target: ')
artifact_dir = input('Artifact directory (e.g. /mnt/dev/margot/artifacts): ')
artifact_dir = Path(artifact_dir)
if not artifact_dir.is_dir():
    print('Artifact directory path not found')
    raise SystemExit

# Example that works
#extract_uuid = '1ef82837-251b-498e-b4f4-72d688e1e614'
#field = 'StatementDate'
#extract_target = 'September 24, 2021'

# Example that fails in finding feature importances
#extract_uuid = 'e6bcb902-87d3-4a15-ab73-6135e5314c6b'
#field = 'E_SalTaxAmt'
#extract_target = '205.23'
#artifact_dir = Path('/mnt/dev/margot/artifacts') 


map_analyzer = MapAnalyzer(extract_uuid, field, artifact_dir, extract_target)
map_analyzer.analyze()