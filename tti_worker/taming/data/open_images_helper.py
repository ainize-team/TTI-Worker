# Code From https://github.com/CompVis/taming-transformers
open_images_unify_categories_for_coco = {
    "/m/03bt1vf": "/m/01g317",
    "/m/04yx4": "/m/01g317",
    "/m/05r655": "/m/01g317",
    "/m/01bl7v": "/m/01g317",
    "/m/0cnyhnx": "/m/01xq0k1",
    "/m/01226z": "/m/018xm",
    "/m/05ctyq": "/m/018xm",
    "/m/058qzx": "/m/04ctx",
    "/m/06pcq": "/m/0l515",
    "/m/03m3pdh": "/m/02crq1",
    "/m/046dlr": "/m/01x3z",
    "/m/0h8mzrc": "/m/01x3z",
}


top_300_classes_plus_coco_compatibility = [
    ("Man", 1060962),
    ("Clothing", 986610),
    ("Tree", 748162),
    ("Woman", 611896),
    ("Person", 610294),
    ("Human face", 442948),
    ("Girl", 175399),
    ("Building", 162147),
    ("Car", 159135),
    ("Plant", 155704),
    ("Human body", 137073),
    ("Flower", 133128),
    ("Window", 127485),
    ("Human arm", 118380),
    ("House", 114365),
    ("Wheel", 111684),
    ("Suit", 99054),
    ("Human hair", 98089),
    ("Human head", 92763),
    ("Chair", 88624),
    ("Boy", 79849),
    ("Table", 73699),
    ("Jeans", 57200),
    ("Tire", 55725),
    ("Skyscraper", 53321),
    ("Food", 52400),
    ("Footwear", 50335),
    ("Dress", 50236),
    ("Human leg", 47124),
    ("Toy", 46636),
    ("Tower", 45605),
    ("Boat", 43486),
    ("Land vehicle", 40541),
    ("Bicycle wheel", 34646),
    ("Palm tree", 33729),
    ("Fashion accessory", 32914),
    ("Glasses", 31940),
    ("Bicycle", 31409),
    ("Furniture", 30656),
    ("Sculpture", 29643),
    ("Bottle", 27558),
    ("Dog", 26980),
    ("Snack", 26796),
    ("Human hand", 26664),
    ("Bird", 25791),
    ("Book", 25415),
    ("Guitar", 24386),
    ("Jacket", 23998),
    ("Poster", 22192),
    ("Dessert", 21284),
    ("Baked goods", 20657),
    ("Drink", 19754),
    ("Flag", 18588),
    ("Houseplant", 18205),
    ("Tableware", 17613),
    ("Airplane", 17218),
    ("Door", 17195),
    ("Sports uniform", 17068),
    ("Shelf", 16865),
    ("Drum", 16612),
    ("Vehicle", 16542),
    ("Microphone", 15269),
    ("Street light", 14957),
    ("Cat", 14879),
    ("Fruit", 13684),
    ("Fast food", 13536),
    ("Animal", 12932),
    ("Vegetable", 12534),
    ("Train", 12358),
    ("Horse", 11948),
    ("Flowerpot", 11728),
    ("Motorcycle", 11621),
    ("Fish", 11517),
    ("Desk", 11405),
    ("Helmet", 10996),
    ("Truck", 10915),
    ("Bus", 10695),
    ("Hat", 10532),
    ("Auto part", 10488),
    ("Musical instrument", 10303),
    ("Sunglasses", 10207),
    ("Picture frame", 10096),
    ("Sports equipment", 10015),
    ("Shorts", 9999),
    ("Wine glass", 9632),
    ("Duck", 9242),
    ("Wine", 9032),
    ("Rose", 8781),
    ("Tie", 8693),
    ("Butterfly", 8436),
    ("Beer", 7978),
    ("Cabinetry", 7956),
    ("Laptop", 7907),
    ("Insect", 7497),
    ("Goggles", 7363),
    ("Shirt", 7098),
    ("Dairy Product", 7021),
    ("Marine invertebrates", 7014),
    ("Cattle", 7006),
    ("Trousers", 6903),
    ("Van", 6843),
    ("Billboard", 6777),
    ("Balloon", 6367),
    ("Human nose", 6103),
    ("Tent", 6073),
    ("Camera", 6014),
    ("Doll", 6002),
    ("Coat", 5951),
    ("Mobile phone", 5758),
    ("Swimwear", 5729),
    ("Strawberry", 5691),
    ("Stairs", 5643),
    ("Goose", 5599),
    ("Umbrella", 5536),
    ("Cake", 5508),
    ("Sun hat", 5475),
    ("Bench", 5310),
    ("Bookcase", 5163),
    ("Bee", 5140),
    ("Computer monitor", 5078),
    ("Hiking equipment", 4983),
    ("Office building", 4981),
    ("Coffee cup", 4748),
    ("Curtain", 4685),
    ("Plate", 4651),
    ("Box", 4621),
    ("Tomato", 4595),
    ("Coffee table", 4529),
    ("Office supplies", 4473),
    ("Maple", 4416),
    ("Muffin", 4365),
    ("Cocktail", 4234),
    ("Castle", 4197),
    ("Couch", 4134),
    ("Pumpkin", 3983),
    ("Computer keyboard", 3960),
    ("Human mouth", 3926),
    ("Christmas tree", 3893),
    ("Mushroom", 3883),
    ("Swimming pool", 3809),
    ("Pastry", 3799),
    ("Lavender (Plant)", 3769),
    ("Football helmet", 3732),
    ("Bread", 3648),
    ("Traffic sign", 3628),
    ("Common sunflower", 3597),
    ("Television", 3550),
    ("Bed", 3525),
    ("Cookie", 3485),
    ("Fountain", 3484),
    ("Paddle", 3447),
    ("Bicycle helmet", 3429),
    ("Porch", 3420),
    ("Deer", 3387),
    ("Fedora", 3339),
    ("Canoe", 3338),
    ("Carnivore", 3266),
    ("Bowl", 3202),
    ("Human eye", 3166),
    ("Ball", 3118),
    ("Pillow", 3077),
    ("Salad", 3061),
    ("Beetle", 3060),
    ("Orange", 3050),
    ("Drawer", 2958),
    ("Platter", 2937),
    ("Elephant", 2921),
    ("Seafood", 2921),
    ("Monkey", 2915),
    ("Countertop", 2879),
    ("Watercraft", 2831),
    ("Helicopter", 2805),
    ("Kitchen appliance", 2797),
    ("Personal flotation device", 2781),
    ("Swan", 2739),
    ("Lamp", 2711),
    ("Boot", 2695),
    ("Bronze sculpture", 2693),
    ("Chicken", 2677),
    ("Taxi", 2643),
    ("Juice", 2615),
    ("Cowboy hat", 2604),
    ("Apple", 2600),
    ("Tin can", 2590),
    ("Necklace", 2564),
    ("Ice cream", 2560),
    ("Human beard", 2539),
    ("Coin", 2536),
    ("Candle", 2515),
    ("Cart", 2512),
    ("High heels", 2441),
    ("Weapon", 2433),
    ("Handbag", 2406),
    ("Penguin", 2396),
    ("Rifle", 2352),
    ("Violin", 2336),
    ("Skull", 2304),
    ("Lantern", 2285),
    ("Scarf", 2269),
    ("Saucer", 2225),
    ("Sheep", 2215),
    ("Vase", 2189),
    ("Lily", 2180),
    ("Mug", 2154),
    ("Parrot", 2140),
    ("Human ear", 2137),
    ("Sandal", 2115),
    ("Lizard", 2100),
    ("Kitchen & dining room table", 2063),
    ("Spider", 1977),
    ("Coffee", 1974),
    ("Goat", 1926),
    ("Squirrel", 1922),
    ("Cello", 1913),
    ("Sushi", 1881),
    ("Tortoise", 1876),
    ("Pizza", 1870),
    ("Studio couch", 1864),
    ("Barrel", 1862),
    ("Cosmetics", 1841),
    ("Moths and butterflies", 1841),
    ("Convenience store", 1817),
    ("Watch", 1792),
    ("Home appliance", 1786),
    ("Harbor seal", 1780),
    ("Luggage and bags", 1756),
    ("Vehicle registration plate", 1754),
    ("Shrimp", 1751),
    ("Jellyfish", 1730),
    ("French fries", 1723),
    ("Egg (Food)", 1698),
    ("Football", 1697),
    ("Musical keyboard", 1683),
    ("Falcon", 1674),
    ("Candy", 1660),
    ("Medical equipment", 1654),
    ("Eagle", 1651),
    ("Dinosaur", 1634),
    ("Surfboard", 1630),
    ("Tank", 1628),
    ("Grape", 1624),
    ("Lion", 1624),
    ("Owl", 1622),
    ("Ski", 1613),
    ("Waste container", 1606),
    ("Frog", 1591),
    ("Sparrow", 1585),
    ("Rabbit", 1581),
    ("Pen", 1546),
    ("Sea lion", 1537),
    ("Spoon", 1521),
    ("Sink", 1512),
    ("Teddy bear", 1507),
    ("Bull", 1495),
    ("Sofa bed", 1490),
    ("Dragonfly", 1479),
    ("Brassiere", 1478),
    ("Chest of drawers", 1472),
    ("Aircraft", 1466),
    ("Human foot", 1463),
    ("Pig", 1455),
    ("Fork", 1454),
    ("Antelope", 1438),
    ("Tripod", 1427),
    ("Tool", 1424),
    ("Cheese", 1422),
    ("Lemon", 1397),
    ("Hamburger", 1393),
    ("Dolphin", 1390),
    ("Mirror", 1390),
    ("Marine mammal", 1387),
    ("Giraffe", 1385),
    ("Snake", 1368),
    ("Gondola", 1364),
    ("Wheelchair", 1360),
    ("Piano", 1358),
    ("Cupboard", 1348),
    ("Banana", 1345),
    ("Trumpet", 1335),
    ("Lighthouse", 1333),
    ("Invertebrate", 1317),
    ("Carrot", 1268),
    ("Sock", 1260),
    ("Tiger", 1241),
    ("Camel", 1224),
    ("Parachute", 1224),
    ("Bathroom accessory", 1223),
    ("Earrings", 1221),
    ("Headphones", 1218),
    ("Skirt", 1198),
    ("Skateboard", 1190),
    ("Sandwich", 1148),
    ("Saxophone", 1141),
    ("Goldfish", 1136),
    ("Stool", 1104),
    ("Traffic light", 1097),
    ("Shellfish", 1081),
    ("Backpack", 1079),
    ("Sea turtle", 1078),
    ("Cucumber", 1075),
    ("Tea", 1051),
    ("Toilet", 1047),
    ("Roller skates", 1040),
    ("Mule", 1039),
    ("Bust", 1031),
    ("Broccoli", 1030),
    ("Crab", 1020),
    ("Oyster", 1019),
    ("Cannon", 1012),
    ("Zebra", 1012),
    ("French horn", 1008),
    ("Grapefruit", 998),
    ("Whiteboard", 997),
    ("Zucchini", 997),
    ("Crocodile", 992),
    ("Clock", 960),
    ("Wall clock", 958),
    ("Doughnut", 869),
    ("Snail", 868),
    ("Baseball glove", 859),
    ("Panda", 830),
    ("Tennis racket", 830),
    ("Pear", 652),
    ("Bagel", 617),
    ("Oven", 616),
    ("Ladybug", 615),
    ("Shark", 615),
    ("Polar bear", 614),
    ("Ostrich", 609),
    ("Hot dog", 473),
    ("Microwave oven", 467),
    ("Fire hydrant", 20),
    ("Stop sign", 20),
    ("Parking meter", 20),
    ("Bear", 20),
    ("Flying disc", 20),
    ("Snowboard", 20),
    ("Tennis ball", 20),
    ("Kite", 20),
    ("Baseball bat", 20),
    ("Kitchen knife", 20),
    ("Knife", 20),
    ("Submarine sandwich", 20),
    ("Computer mouse", 20),
    ("Remote control", 20),
    ("Toaster", 20),
    ("Sink", 20),
    ("Refrigerator", 20),
    ("Alarm clock", 20),
    ("Wall clock", 20),
    ("Scissors", 20),
    ("Hair dryer", 20),
    ("Toothbrush", 20),
    ("Suitcase", 20),
]
