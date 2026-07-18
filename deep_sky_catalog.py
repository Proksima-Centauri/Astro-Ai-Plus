"""Unified deep-sky catalog used by Astro Ai object matching.

This module contains curated entries with coordinates for popular deep-sky
objects and serves as a local offline fallback when remote lookup is
unavailable.
"""

_MESSIER_DATA = """
M1|Crab Nebula|NGC1952,Taurus A|83.6331|22.0145
M2|NGC7089|NGC7089|323.3625|-0.8233
M3|NGC5272|NGC5272|205.5484|28.3773
M4|NGC6121|NGC6121|245.8968|-26.5257
M5|NGC5904|NGC5904|229.6384|2.0810
M6|Butterfly Cluster|NGC6405|265.0833|-32.2167
M7|Ptolemy Cluster|NGC6475|268.4625|-34.7928
M8|Lagoon Nebula|NGC6523|270.9210|-24.3802
M9|NGC6333|NGC6333|259.7976|-18.5163
M10|NGC6254|NGC6254|254.2877|-4.1003
M11|Wild Duck Cluster|NGC6705|282.7704|-6.2708
M12|NGC6218|NGC6218|251.8091|-1.9485
M13|Hercules Globular Cluster|NGC6205|250.4235|36.4613
M14|NGC6402|NGC6402|264.4004|-3.2459
M15|Pegasus Cluster|NGC7078|322.4930|12.1670
M16|Eagle Nebula|NGC6611|274.7000|-13.8067
M17|Omega Nebula|NGC6618,Swan Nebula|275.1963|-16.1716
M18|NGC6613|NGC6613|274.9958|-17.1347
M19|NGC6273|NGC6273|255.6575|-26.2679
M20|Trifid Nebula|NGC6514|270.6750|-23.0300
M21|NGC6531|NGC6531|271.0542|-22.4900
M22|Sagittarius Cluster|NGC6656|279.0998|-23.9048
M23|NGC6494|NGC6494|269.2667|-18.9850
M24|Sagittarius Star Cloud|IC4715|274.2250|-18.4833
M25|IC4725|IC4725|277.9458|-19.1167
M26|NGC6694|NGC6694|281.3250|-9.3867
M27|Dumbbell Nebula|NGC6853|299.9016|22.7210
M28|NGC6626|NGC6626|276.1364|-24.8698
M29|Cooling Tower Cluster|NGC6913|305.9833|38.5233
M30|NGC7099|NGC7099|325.0922|-23.1799
M31|Andromeda Galaxy|NGC224|10.6847|41.2690
M32|Le Gentil|NGC221|10.6743|40.8652
M33|Triangulum Galaxy|NGC598|23.4621|30.6602
M34|Spiral Cluster|NGC1039|40.5208|42.7833
M35|Shoe-Buckle Cluster|NGC2168|92.2250|24.3333
M36|Pinwheel Cluster|NGC1960|84.0750|34.1400
M37|Salt and Pepper Cluster|NGC2099|88.0750|32.5533
M38|Starfish Cluster|NGC1912|82.1792|35.8550
M39|NGC7092|NGC7092|322.9500|48.4333
M40|Winnecke 4|WNC4|185.5529|58.0831
M41|Little Beehive Cluster|NGC2287|101.5042|-20.7567
M42|Orion Nebula|NGC1976|83.8221|-5.3911
M43|De Mairan's Nebula|NGC1982|83.8792|-5.2700
M44|Beehive Cluster|NGC2632,Praesepe|130.0250|19.6667
M45|Pleiades|Melotte22,Seven Sisters|56.7500|24.1167
M46|NGC2437|NGC2437|115.4417|-14.8100
M47|NGC2422|NGC2422|114.1458|-14.4833
M48|NGC2548|NGC2548|123.4333|-5.8000
M49|NGC4472|NGC4472|187.4446|8.0004
M50|Heart-Shaped Cluster|NGC2323|105.8000|-8.3378
M51|Whirlpool Galaxy|NGC5194|202.4696|47.1952
M52|Scorpion Cluster|NGC7654|351.2000|61.5933
M53|NGC5024|NGC5024|198.2302|18.1682
M54|NGC6715|NGC6715|283.7639|-30.4799
M55|Summer Rose Star|NGC6809|294.9988|-30.9647
M56|NGC6779|NGC6779|289.1479|30.1845
M57|Ring Nebula|NGC6720|283.3962|33.0292
M58|NGC4579|NGC4579|189.4316|11.8181
M59|NGC4621|NGC4621|190.5097|11.6469
M60|NGC4649|NGC4649|190.9167|11.5527
M61|Swelling Spiral Galaxy|NGC4303|185.4789|4.4737
M62|Flickering Globular|NGC6266|255.3025|-30.1124
M63|Sunflower Galaxy|NGC5055|198.9555|42.0293
M64|Black Eye Galaxy|NGC4826|194.1821|21.6827
M65|Leo Triplet|NGC3623|169.7332|13.0922
M66|Leo Triplet|NGC3627|170.0625|12.9915
M67|King Cobra Cluster|NGC2682|132.8250|11.8167
M68|NGC4590|NGC4590|189.8666|-26.7441
M69|NGC6637|NGC6637|277.8463|-32.3481
M70|NGC6681|NGC6681|280.8032|-32.2921
M71|Angelfish Cluster|NGC6838|298.4437|18.7792
M72|NGC6981|NGC6981|313.3654|-12.5373
M73|NGC6994|NGC6994|314.7421|-12.6346
M74|Phantom Galaxy|NGC628|24.1741|15.7835
M75|NGC6864|NGC6864|301.5202|-21.9217
M76|Little Dumbbell Nebula|NGC650,NGC651|25.5821|51.5755
M77|Cetus A|NGC1068|40.6696|-0.0133
M78|NGC2068|NGC2068|86.6704|0.0792
M79|NGC1904|NGC1904|81.0462|-24.5243
M80|NGC6093|NGC6093|244.2600|-22.9761
M81|Bode's Galaxy|NGC3031|148.8882|69.0653
M82|Cigar Galaxy|NGC3034|148.9685|69.6797
M83|Southern Pinwheel Galaxy|NGC5236|204.2538|-29.8658
M84|NGC4374|NGC4374|186.2656|12.8869
M85|NGC4382|NGC4382|186.3502|18.1911
M86|NGC4406|NGC4406|186.5480|12.9462
M87|Virgo A|NGC4486|187.7059|12.3911
M88|NGC4501|NGC4501|188.0000|14.4204
M89|NGC4552|NGC4552|188.9159|12.5563
M90|NGC4569|NGC4569|189.2076|13.1629
M91|NGC4548|NGC4548|188.8601|14.4963
M92|NGC6341|NGC6341|259.2808|43.1365
M93|NGC2447|NGC2447|116.1500|-23.8567
M94|Croc's Eye Galaxy|NGC4736|192.7211|41.1202
M95|NGC3351|NGC3351|160.9906|11.7037
M96|NGC3368|NGC3368|161.6906|11.8199
M97|Owl Nebula|NGC3587|168.6988|55.0190
M98|NGC4192|NGC4192|183.4513|14.9005
M99|Coma Pinwheel Galaxy|NGC4254|184.7062|14.4165
M100|Mirror Galaxy|NGC4321|185.7287|15.8223
M101|Pinwheel Galaxy|NGC5457|210.8023|54.3489
M102|Spindle Galaxy|NGC5866|226.6221|55.7633
M103|NGC581|NGC581|23.3375|60.6583
M104|Sombrero Galaxy|NGC4594|189.9976|-11.6231
M105|NGC3379|NGC3379|161.9567|12.5816
M106|NGC4258|NGC4258|184.7396|47.3038
M107|NGC6171|NGC6171|248.1328|-13.0538
M108|Surfboard Galaxy|NGC3556|167.8790|55.6741
M109|Vacuum Cleaner Galaxy|NGC3992|179.3999|53.3745
M110|Edward Young Star|NGC205|10.0919|41.6853
"""


def _parse_messier_catalog():
    catalog = []
    for line in _MESSIER_DATA.strip().splitlines():
        name, display_name, aliases, ra, dec = line.split("|")
        catalog.append({
            "name": name,
            "display_name": display_name,
            "aliases": [alias.strip() for alias in aliases.split(",") if alias.strip()],
            "ra": float(ra),
            "dec": float(dec),
            "catalog": "Messier",
        })
    return catalog


MESSIER_CATALOG = _parse_messier_catalog()

_EXTRA_DATA = """
NGC1499|California Nebula|Sh2-220,Caldwell31|60.0000|36.6167|NGC
NGC1973|Running Man Nebula|NGC1975,NGC1977,Sh2-279|83.8167|-4.8500|NGC
NGC2024|Flame Nebula|Sh2-277|85.4275|-1.8483|NGC
NGC2237|Rosette Nebula|NGC2238,NGC2239,NGC2246,Sh2-275,Caldwell49|97.9917|4.9417|NGC
NGC2244|Rosette Cluster|Caldwell50|97.9500|4.9500|NGC
NGC2264|Cone Nebula Cluster|Christmas Tree Cluster,Sh2-273|100.2420|9.8950|NGC
NGC2359|Thor's Helmet|Sh2-298|109.1333|-13.2167|NGC
NGC2392|Eskimo Nebula|Clown Face Nebula,Caldwell39|112.2946|20.9118|NGC
NGC2403|Caldwell7|C7|114.2142|65.6026|NGC
NGC281|Pacman Nebula|Sh2-184,IC11|13.0083|56.6167|NGC
NGC2903|Caldwell56|C56|143.0421|21.5008|NGC
NGC3190|Hickson 44|HCG44|154.0129|21.8323|NGC
NGC3372|Carina Nebula|Eta Carinae Nebula,Caldwell92|161.2650|-59.8667|NGC
NGC3628|Hamburger Galaxy|Leo Triplet|170.0708|13.5894|NGC
NGC3718|Arp214|Arp 214|173.1458|53.0678|NGC
NGC4038|Antennae Galaxies|NGC4039,Caldwell60,Caldwell61|180.4704|-18.8676|NGC
NGC4216|Silver Streak Galaxy|Caldwell38|183.9762|13.1494|NGC
NGC4565|Needle Galaxy|Caldwell38|189.0866|25.9878|NGC
NGC4631|Whale Galaxy|Caldwell32|190.5334|32.5415|NGC
NGC4656|Hockey Stick Galaxy|NGC4657|190.9900|32.1692|NGC
NGC4725|Caldwell45|C45|192.6108|25.5008|NGC
NGC5139|Omega Centauri|Caldwell80|201.6970|-47.4795|NGC
NGC5128|Centaurus A|Caldwell77|201.3651|-43.0191|NGC
NGC5907|Splinter Galaxy|Knife Edge Galaxy|228.9740|56.3288|NGC
NGC6188|Fighting Dragons Nebula|Rim Nebula|250.6125|-48.7667|NGC
NGC6334|Cat's Paw Nebula|Bear Claw Nebula|258.1979|-35.8578|NGC
NGC6357|Lobster Nebula|War and Peace Nebula|261.2833|-34.2000|NGC
NGC6543|Cat's Eye Nebula|Caldwell6|269.6392|66.6331|NGC
NGC6888|Crescent Nebula|Caldwell27,Sh2-105|305.2500|38.3500|NGC
NGC6960|Western Veil Nebula|Witch's Broom,Caldwell34|311.2917|30.7167|NGC
NGC6992|Eastern Veil Nebula|Caldwell33|312.5000|31.7167|NGC
NGC6995|Network Nebula|Veil Nebula|312.9167|31.2167|NGC
NGC7000|North America Nebula|Caldwell20,Sh2-117|314.7500|44.3333|NGC
NGC7008|Fetus Nebula|Caldwell38|315.3479|54.5433|NGC
NGC7023|Iris Nebula|Caldwell4|316.7500|68.1667|NGC
NGC7293|Helix Nebula|Caldwell63|337.4100|-20.8370|NGC
NGC7380|Wizard Nebula|Sh2-142|342.5125|58.1000|NGC
NGC7635|Bubble Nebula|Caldwell11,Sh2-162|350.2000|61.2000|NGC
NGC7822|Question Mark Nebula|Cederblad214,Sh2-171|0.9833|67.1667|NGC
IC10|Starburst Galaxy|Caldwell63|5.0721|59.3039|IC
IC59|Gamma Cassiopeiae Nebula|Sh2-185|14.6333|61.1333|IC
IC63|Ghost of Cassiopeia|Sh2-185|14.7500|60.9167|IC
IC1396|Elephant Trunk Nebula|IC1396A,Sh2-131|324.7000|57.5000|IC
IC1613|Caldwell51|C51|16.1992|2.1178|IC
IC1795|Fish Head Nebula|Northern Bear Nebula|40.3250|61.9833|IC
IC1805|Heart Nebula|Sh2-190|38.5000|61.4500|IC
IC1848|Soul Nebula|Embryo Nebula,Sh2-199|43.1500|60.4333|IC
IC2118|Witch Head Nebula|Caldwell38|75.0833|-7.2167|IC
IC2177|Seagull Nebula|Sh2-296|105.0000|-10.7000|IC
IC342|Hidden Galaxy|Caldwell5|56.7033|68.0961|IC
IC405|Flaming Star Nebula|Sh2-229,Caldwell31|79.0750|34.4667|IC
IC410|Tadpoles Nebula|Sh2-236|80.4167|33.4167|IC
IC443|Jellyfish Nebula|Sh2-248|94.2750|22.6500|IC
IC4592|Blue Horsehead Nebula|Caldwell102|240.4375|-19.4500|IC
IC4628|Prawn Nebula|Gum56|249.9500|-40.4500|IC
IC5070|Pelican Nebula|Sh2-117|312.7500|44.3667|IC
IC5146|Cocoon Nebula|Caldwell19,Sh2-125|328.3667|47.2667|IC
Sh2-54|Serpens Cloud|RCW167|275.0000|-11.7000|Sharpless
Sh2-64|Serpens South|LBN90|277.3000|-2.0500|Sharpless
Sh2-86|NGC6820 Nebula|Vulpecula OB1|295.2000|23.3000|Sharpless
Sh2-101|Tulip Nebula|LBN168|304.9500|35.2500|Sharpless
Sh2-106|Snow Angel Nebula|S106|306.8000|37.3667|Sharpless
Sh2-112|Emission Nebula|LBN337|306.5500|45.6500|Sharpless
Sh2-115|Emission Nebula|LBN357|309.3833|46.8833|Sharpless
Sh2-119|Clamshell Nebula|LBN391|316.2000|43.9000|Sharpless
Sh2-126|Lacerta Nebula|LBN437|332.0000|40.9000|Sharpless
Sh2-129|Flying Bat Nebula|Ou4|318.0000|59.9000|Sharpless
Sh2-132|Lion Nebula|LBN473|335.0000|56.0000|Sharpless
Sh2-135|Emission Nebula|LBN492|337.7000|58.4000|Sharpless
Sh2-140|Cepheus Nebula|LBN505|333.7000|63.3000|Sharpless
Sh2-150|Emission Nebula|LBN536|343.5000|60.9500|Sharpless
Sh2-155|Cave Nebula|Caldwell9,LBN529|333.9167|62.5167|Sharpless
Sh2-157|Lobster Claw Nebula|LBN537|350.2500|60.2500|Sharpless
Sh2-158|NGC7538|Caldwell38|348.2500|61.4667|Sharpless
Sh2-170|Little Rosette Nebula|LBN577|0.4000|64.6500|Sharpless
Sh2-171|Cederblad 214|NGC7822|0.9833|67.1667|Sharpless
Sh2-188|Dolphin Head Nebula|PK128-04.1|20.1500|58.4333|Sharpless
Sh2-199|Soul Nebula|IC1848|43.1500|60.4333|Sharpless
Sh2-216|Ancient Planetary Nebula|PK158+00.1|70.9500|46.7333|Sharpless
Sh2-220|California Nebula|NGC1499|60.0000|36.6167|Sharpless
Sh2-224|Supernova Remnant|SNR156.2+5.7|75.3000|42.7000|Sharpless
Sh2-240|Simeis 147|Spaghetti Nebula|82.2500|28.3000|Sharpless
Sh2-261|Lower's Nebula|LBN863|92.5000|15.7000|Sharpless
Sh2-264|Lambda Orionis Ring|Angelfish Nebula|83.8000|9.9000|Sharpless
Sh2-274|Medusa Nebula|Abell21|112.2500|13.2500|Sharpless
Sh2-275|Rosette Nebula|NGC2237|97.9917|4.9417|Sharpless
Sh2-276|Barnard's Loop|LBN962|83.0000|-4.6000|Sharpless
Sh2-279|Running Man Nebula|NGC1977|83.8167|-4.8500|Sharpless
Sh2-284|Emission Nebula|LBN983|102.7500|0.1000|Sharpless
Sh2-290|Abell 31|PK219+31.1|130.2250|8.8833|Sharpless
Sh2-308|Dolphin Head Nebula|RCW11|103.0000|-23.9167|Sharpless
Sh2-310|Lambda Centauri Nebula|IC2944|174.0000|-63.3500|Sharpless
B33|Horsehead Nebula|Barnard33,IC434|85.2458|-2.4583|Barnard
B72|Snake Nebula|Barnard72|259.6250|-23.6333|Barnard
B86|Ink Spot Nebula|Barnard86|270.6083|-27.9833|Barnard
B168|Dark Cigar Nebula|Barnard168|326.7500|47.5000|Barnard
LDN673|Dark Nebula|LBN|286.5000|11.0000|LDN
LDN1235|Dark Shark Nebula|VdB152|337.5000|70.2500|LDN
LDN1251|Dark Nebula|Cepheus Flare|339.4000|75.1500|LDN
LBN331|North America Complex|Cygnus Wall|314.7500|44.3333|LBN
LBN437|Gecko Nebula|Sh2-126|332.0000|40.9000|LBN
C4|Iris Nebula|NGC7023|316.7500|68.1667|Caldwell
C5|Hidden Galaxy|IC342|56.7033|68.0961|Caldwell
C6|Cat's Eye Nebula|NGC6543|269.6392|66.6331|Caldwell
C9|Cave Nebula|Sh2-155|333.9167|62.5167|Caldwell
C11|Bubble Nebula|NGC7635|350.2000|61.2000|Caldwell
C19|Cocoon Nebula|IC5146|328.3667|47.2667|Caldwell
C20|North America Nebula|NGC7000|314.7500|44.3333|Caldwell
C27|Crescent Nebula|NGC6888|305.2500|38.3500|Caldwell
C31|California Nebula|NGC1499|60.0000|36.6167|Caldwell
C33|Eastern Veil Nebula|NGC6992|312.5000|31.7167|Caldwell
C34|Western Veil Nebula|NGC6960|311.2917|30.7167|Caldwell
C49|Rosette Nebula|NGC2237|97.9917|4.9417|Caldwell
C63|Helix Nebula|NGC7293|337.4100|-20.8370|Caldwell
"""


def _parse_extra_catalog():
    catalog = []
    for line in _EXTRA_DATA.strip().splitlines():
        name, display_name, aliases, ra, dec, catalog_name = line.split("|")
        catalog.append({
            "name": name,
            "display_name": display_name,
            "aliases": [alias.strip() for alias in aliases.split(",") if alias.strip()],
            "ra": float(ra),
            "dec": float(dec),
            "catalog": catalog_name,
        })
    return catalog


DEEP_SKY_CATALOG = []
_seen_names = set()
for _obj in list(MESSIER_CATALOG) + _parse_extra_catalog():
    _copy = dict(_obj)
    _copy.setdefault("catalog", "Messier")
    _key = _copy["name"].upper()
    if _key in _seen_names:
        continue
    _seen_names.add(_key)
    DEEP_SKY_CATALOG.append(_copy)

