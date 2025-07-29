# import numpy as np
# import random
# class GenBoard:
#     def __init__(self, layout: str = '', ring: int = 2):
#         if layout != '':
#             self.board_layout = layout
#         else:
#             self.board_layout = random.choice(self.generate_list_of_all_possible_boards())
#         self.values = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]
#         self.tile_index_to_tile_coordinate = {
#             index:value for index,value in enumerate([(0,0)]+[(1,n) for n in range(6)] + [(2,m) for m in range(12)])
#             }
#         self.tile_coordinate_to_tile_index = {v:k for k,v in self.tile_index_to_tile_coordinate.items()}

#         self.node_index_to_node_coordinate = {
#             index:value for index,value in enumerate([(0,n) for n in range(6)] + [(1,m) for m in range(18)] + [(2,k) for k in range(30)])
#             }
#         self.node_coordinate_to_node_index = {v:k for k,v in  self.node_index_to_node_coordinate.items()}

#         self.nodes_to_neighbour_nodes = self.calculate_nodes_to_nodes()
#         self.nodes_to_neighbour_tiles = self.calculate_nodes_to_tiles()

#         self.edge_index_to_edge_coordinate = {
#             index:value for index,value in enumerate(self.calculate_unique_edges())
#             }
#         self.edge_coordinate_to_edge_index = {v:k for k,v in  self.edge_index_to_edge_coordinate.items()}
#         self.tile_coordinate_to_frequency = self.calculate_tile_coordinate_to_frequency()
#         self.earning_values_per_node = self.calculate_earning_value_per_node()
#         self.dice_effect_per_node_vnk= self.calculate_dice_effect_for_node()

#     def calculate_tile_coordinate_to_frequency(self):
#         frequencies = [ sum([ sum([1 for a in range(1,7) if a+b == n]) for b in range(1,7) ]) for n in self.values ]
#         coord_order = [(2,m) for m in range(12)] + [(1,n) for n in range(6)]
#         tile_coordinate_to_frequency = {a:b for a,b in zip(coord_order,frequencies)}
#         tile_coordinate_to_frequency.update({(0,0):0})
#         return tile_coordinate_to_frequency
    
#     def calculate_nodes_to_nodes(self):
#         nodes_to_nodes = dict([])
#         for index in range(6):
#             nodes_to_nodes.update( {(0,index):[(0,(index-1)%6),(0,(index+1)%6),(1,(-1+3*index)%18)]})
#         for index in range(18):
#             q,r = divmod(index,3)
#             if r == 0:
#                 spoke = (2,(5*q)%30)
#             elif r == 1:
#                 spoke = (2,(5*q+3)%30)
#             else:
#                 spoke = (0,(q+1)%6)
#             nodes_to_nodes.update( {(1,index):[(1,(index-1)%18),(1,(index+1)%18),spoke]})
#         for index in range(30):
#             q,r = divmod(index,5)
#             if r == 0:
#                 spoke = (1,(3*q)%18)
#                 nodes_to_nodes.update( {(2,index):[(2,(index-1)%30),(2,(index+1)%30),spoke]})
#             elif r == 3:
#                 spoke = (1,(3*q+1)%18)
#                 nodes_to_nodes.update( {(2,index):[(2,(index-1)%30),(2,(index+1)%30),spoke]})
#             else:
#                 nodes_to_nodes.update( {(2,index):[(2,(index-1)%30),(2,(index+1)%30)]})
#         return nodes_to_nodes
    
#     def calculate_nodes_to_tiles(self):
#         # for every node map the neighboring tiles
#         nodes_to_tiles = dict([])
#         # first_row_of_nodes
#         nodes_to_tiles.update( { (0,0) : [(1,5),(1,0)]})
#         for index in range(1,6):
#             nodes_to_tiles.update( { (0,index) : [(1,index-1),(1,index),(0,0)]})
#         # second_row_of_nodes
#         nodes_to_tiles.update( { (1,0) : [(2,11),(2,0),(1,0)]})
#         nodes_to_tiles.update( { (1,17) : [(2,11),(1,5),(1,0)]})
#         for index in [3,6,9,12,15]:
#             spoke = (index)//3
#             nodes_to_tiles.update( { (1,index) : [(2,(2*spoke)-1),(2,2*spoke),(1,spoke)]})
#         for index in [1,4,7,10,13,16]:
#             spoke = (index)//3
#             nodes_to_tiles.update( { (1,index) : [(2,(2*spoke)+1),(2,2*spoke),(1,spoke)]})
#         for index in [2,5,8,11,14]:
#             spoke = (index)//3
#             nodes_to_tiles.update( { (1,index) : [(2,(2*spoke)+1),(1,spoke),(1,spoke+1)]})
#         # third_row_of_nodes
#         nodes_to_tiles.update( {(2,0) : [(2,0),(2,11)]})
#         for index in range(1,30):
#             q,r = divmod(index,5)
#             if r == 0:
#                 nodes_to_tiles.update( { (2,index) : [(2,2*q),(2,2*q-1)]})
#             elif r in [1,2]:
#                 nodes_to_tiles.update( { (2,index) : [(2,2*q)]})
#             elif r == 3:
#                 nodes_to_tiles.update( { (2,index) : [(2,2*q),(2,2*q+1)]})
#             else:
#                 nodes_to_tiles.update( { (2,index) : [(2,2*q+1)]})
#         return nodes_to_tiles
    
#     def calculate_unique_edges(self):
#         # list of unique edges
#         unique_edges = []
#         for node_1,connections in self.nodes_to_neighbour_nodes.items():
#             for node_2 in connections:
#                 if (node_1,node_2) not in unique_edges and (node_2,node_1) not in unique_edges:
#                     unique_edges.append((node_1,node_2))
#         return unique_edges
    
#     def edge_to_unique_edge(self, edge):
#         node_1,node_2 = edge
#         if node_1[0] < node_2[0]:
#             return (node_1,node_2)
#         elif node_1[0] > node_2[0]:
#             return (node_2,node_1)
#         elif node_1[1] < node_2[1]:
#             return (node_1,node_2)
#         else:
#             return (node_2,node_1)
    

#     def generate_list_of_all_possible_boards(self):
#         #For first ring number of combinations:
#         # sequence of 6 without same twice, including closing circle
#         tiles = {'S':4,'W':4,'G':4, 'O': 3, 'B':3}
#         boards = [tile for tile in tiles.keys()]
#         for _ in range(5):
#             new_boards = []
#             for board in boards:
#                 for tile in tiles.keys():
#                     if len(board) < 5:
#                         if tile != board[-1] and board.count(tile) < tiles[tile]:
#                             new_boards.append(board + tile)
#                     else:
#                         if tile != board[-1] and tile != board[0] and board.count(tile) < tiles[tile]:
#                             new_boards.append(board + tile)
#             boards = new_boards

#         # only unique permutations
#         def permutations(s):
#             return [s[n:] + s[:n] for n in range(len(s))]

#         uniques = []
#         for board in boards:
#             for p in permutations(board):
#                 if p in uniques:
#                     break
#             else:
#                 uniques.append(board)

#         # add second row
#         full_boards = []
#         for p in uniques:
#             for index in range(12):
#                 if index == 0 and p.count(tile) < tiles[tile]:
#                     rings = [tile for tile in tiles if tile != p[0]]
#                 elif index in [1,3,7,9]:
#                     new_rings = []
#                     for ring in rings:
#                         for tile in tiles:
#                             if tile == ring[-1]:
#                                 continue
#                             if tile == p[index//2] or tile == p[(index//2) + 1]:
#                                 continue
#                             if ring.count(tile) + p.count(tile) >= tiles[tile]:
#                                 continue
#                             new_rings.append(ring + tile)
#                     rings = new_rings
#                 elif index in [2,4,6,8]:
#                     new_rings = []
#                     for ring in rings:
#                         for tile in tiles:
#                             if tile == ring[-1]:
#                                 continue
#                             if tile == p[index//2]:
#                                 continue
#                             if ring.count(tile) + p.count(tile) >= tiles[tile]:
#                                 continue
#                             new_rings.append(ring + tile)
#                     rings = new_rings
#                 else:
#                     new_rings = []
#                     for ring in rings:
#                         for tile in tiles:
#                             if tile == ring[0] or tile == ring[-1]:
#                                 continue
#                             if tile == p[0] or tile == p[-1]:
#                                 continue
#                             if ring.count(tile) + p.count(tile) >= tiles[tile]:
#                                 continue
#                             new_rings.append(ring + tile)
#                     rings = new_rings
#             for ring in rings:
#                 full_boards.append( ("D",p,ring))

#         return [b[0] + b[1] + b[2] for b in full_boards]
    
#     def calculate_earning_value_per_node(self):
#         earning_values = { node:{'S':0,'W':0,'G':0, 'O': 0, 'B':0} for node in self.node_coordinate_to_node_index.keys()}
#         for node,nb_tiles in self.nodes_to_neighbour_tiles.items():
#             for tile in nb_tiles:
#                 if tile == (0,0):
#                     continue
#                 type = self.board_layout[self.tile_coordinate_to_tile_index[tile]]
#                 frequency = self.tile_coordinate_to_frequency[tile]
#                 earning_values[node][type] += frequency
#         return earning_values
    
#     def calculate_dice_effect_for_node(self):

#         coord_order = [(2,m) for m in range(12)] + [(1,n) for n in range(6)]
#         tile_coordinate_to_value = {a:b for a,b in zip(coord_order,self.values)}
#         tile_coordinate_to_value.update({(0,0):0})
#         tile_coordinate_to_type = {a:b for a,b in zip(self.tile_coordinate_to_tile_index.keys(),self.board_layout)}

#         effect_of_dice_value_for_nodes = {v:{n:{'D':0,'S':0,'W':0,'G':0, 'O': 0, 'B':0} for n in self.node_coordinate_to_node_index.keys()} for v in self.values + [0]}   
#         for tile,type in tile_coordinate_to_value.items():
#             value = tile_coordinate_to_value[tile]
#             type = tile_coordinate_to_type[tile]
#             for n, nb_tiles in self.nodes_to_neighbour_tiles.items():
#                 if tile in nb_tiles:
#                     effect_of_dice_value_for_nodes[value][n][type] += 1
            
#         return effect_of_dice_value_for_nodes