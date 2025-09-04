"""
DO NOT TOUCH
The embedding_driver automatically copies this file it to the Raspberry Pi at startup, populates it, 
and uses it to configure the SLM. 
"""

max_cells_W = int((slm_W ) / (N + buffer_space))
max_cells_H = int((slm_H + buffer_space) / (N + buffer_space))

if localization_N is not None:
    max_cells = max_cells_H * max_cells_W - 4*localization_N**2
    num_pilot_cells = max_cells_W * 2 + max_cells_H * 2 - 4 - ((2*localization_N - 1)*4)
    num_info_cells = (max_cells_H * max_cells_W) - num_pilot_cells - 4*(localization_N**2)
    reserved_localization_cells = []
else:
    max_cells = max_cells_H * max_cells_W 
    num_pilot_cells = max_cells_W * 2 + max_cells_H * 2 - 4
    num_info_cells = (max_cells_H * max_cells_W) - num_pilot_cells
    reserved_localization_cells = [] #"r-c" index format. leave empty if don't want any reserved.
    count = 0
    for c in range(max_cells_W):
        if count % 2 == 0:
            reserved_localization_cells.append(f"{0}-{c}")
        count += 1
    for r in range(1, max_cells_H):
        if count % 2 == 0:
            reserved_localization_cells.append(f"{r}-{max_cells_W-1}")
        count += 1
    for c in range(max_cells_W-2, -1, -1):
        if count % 2 == 0:
            reserved_localization_cells.append(f"{max_cells_H-1}-{c}")
        count += 1
    for r in range(max_cells_H-2, 0, -1):
        if count % 2 == 0:
            reserved_localization_cells.append(f"{r}-0")
        count += 1

