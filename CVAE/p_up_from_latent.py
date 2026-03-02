from apply_probing import apply_probing

# mapping is chromosome 0 - 23 instead of 1-24. The list stored is a list of the
# the latent dims we wanna probe and the value we wanna probe them to 
# ranging from 0-31 instead of 1-32 and the second element is what we want to probe 
# that latent dimension to.
centromere_dim_mapping = ["chrom0",[(0,-3),(6,-3),(5,-3)],
                            "chrom1",[(2,-3),(11,-3),(20,-3)],
                            "chrom2",[(2,-3),(7,-3),(9,-3)],
                            "chrom3",[(3,3),(30,-3),(30,3)],
                            "chrom4",[(0,-3),(2,-3),(20,-3)],# this one im guessing!
                            "chrom5",[(0,-3),(3,-3),(26,3)],
                            "chrom6",[(3,-3),(7,3),(31,-3)],
                            "chrom7",[(4,-3),(18,-3),(21,3)],
                            "chrom8",[(5,3),(9,3),(26,-3)],
                            "chrom9",[(7,-3),(8,-3),(16,-3)],
                            "chrom10",[(0,3),(9,3),(11,-3)],
                            "chrom11",[(6,3),(7,3),(10,-3)],
                            "chrom12",[(7,-3),(8,-3),(15,3)],
                            "chrom13",[(11,-3),(21,-3),(27,-3)],
                            "chrom14",[(11,-3),(21,-3),(27,-3)], # this one is random!
                            "chrom15",[(0,-3),(14,3),(27,3)], 
                            "chrom16",[(0,-3),(14,3),(27,3)],# this one is random!
                            "chrom17",[(0,-3),(14,3),(27,3)],# this one is random!
                            "chrom18",[(0,-3),(14,3),(27,3)], # this one is random!
                            "chrom19",[(0,-3),(14,3),(27,3)], # this one is random!
                            "chrom20",[(0,-3),(14,3),(27,3)],# this one is random!
                            "chrom21",[(0,-3),(14,3),(27,3)],# this one is random!
                            "chrom22",[(0,-3),(14,3),(27,3)],# this one is random!
                            "chromX", [(0,-3),(14,3),(27,3)],  # this one is random
]

