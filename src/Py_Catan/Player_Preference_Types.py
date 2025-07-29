from Py_Catan.Preferences import PlayerPreferences
import numpy as np

d = {'full_score': 1000.0, 'streets': 10, 'villages': 20, 'towns': 20,
    'penalty_reference_for_too_many_cards': 1.0, 'cards_in_hand': 1, 'hand_for_street':0, 'hand_for_village': 0,
    'hand_for_town': 0, 'street_build_options': 0, 'village_build_options': 0, 'cards_earning_power': 0,
        'hand_for_street_missing_one': 0, 'hand_for_village_missing_one': 0, 'hand_for_town_missing_one': 0,
        'secondary_village_build_options': 0, 'direct_options_earning_power': 2, 'hand_for_village_missing_two': 0,
    'hand_for_town_missing_two': 0, 'secondary_options_earning_power': 0,
    'resource_type_weight': np.array([0, 0 , 1, 2, 1, 2])}
pref_1 = PlayerPreferences(**d).normalized()

d = {'full_score': 500.0, 'streets': 0, 'villages': 0, 'towns': 0,
    'penalty_reference_for_too_many_cards': 1.0, 'cards_in_hand': 0, 'hand_for_street':0, 'hand_for_village': 0,
    'hand_for_town': 0, 'street_build_options': 0.1, 'village_build_options': 0, 'cards_earning_power': 0,
        'hand_for_street_missing_one': 0, 'hand_for_village_missing_one': 0, 'hand_for_town_missing_one': 0,
        'secondary_village_build_options': 0, 'direct_options_earning_power': 0, 'hand_for_village_missing_two': 0,
    'hand_for_town_missing_two': 0, 'secondary_options_earning_power':0,
    'resource_type_weight': np.array([0.1, 0. , 0.3, 0.3, 0.2, 0.1])}
pref_2 = PlayerPreferences(**d).normalized()

d = {'full_score': 1000.0, 'streets': 10, 'villages': 20, 'towns': 20,
    'penalty_reference_for_too_many_cards': 3.0, 'cards_in_hand': 0.1, 'hand_for_street':2, 'hand_for_village': 2,
    'hand_for_town': 2, 'street_build_options': 0, 'village_build_options': 0.1, 'cards_earning_power': 5,
        'hand_for_street_missing_one': 0.2, 'hand_for_village_missing_one': 0.2, 'hand_for_town_missing_one': 0.2,
        'secondary_village_build_options': 0, 'direct_options_earning_power': 2, 'hand_for_village_missing_two': 0.1,
    'hand_for_town_missing_two': 0.1, 'secondary_options_earning_power': 1,
    'resource_type_weight': np.array([2, 0. , 1, 1, 1, 2])}
pref_3 = PlayerPreferences(**d).normalized()

d = {'full_score': 500.0, 'streets': 10, 'villages': 10, 'towns': 20,
    'penalty_reference_for_too_many_cards': 3.0, 'cards_in_hand': 0.1, 'hand_for_street':2, 'hand_for_village': 2,
    'hand_for_town': 2, 'street_build_options': 0.1, 'village_build_options': 0.1, 'cards_earning_power': 0.2,
        'hand_for_street_missing_one': 0, 'hand_for_village_missing_one': 0, 'hand_for_town_missing_one': 0,
        'secondary_village_build_options': 0, 'direct_options_earning_power': 0.1, 'hand_for_village_missing_two': 0.1,
    'hand_for_town_missing_two': 1, 'secondary_options_earning_power':1,
    'resource_type_weight': np.array([0.1, 0. , 0.3, 0.3, 0.2, 0.1])}
pref_4 = PlayerPreferences(**d).normalized()

d = {'full_score': 500.0, 'streets': 0.12656033870946917, 'villages': 0.2698917771028186, 'towns': 0.2484819241140178, 'penalty_reference_for_too_many_cards': 7.0, 'cards_in_hand': 0.0014212937980396605, 'hand_for_street': 0.02955298336887335, 'hand_for_village': 0.03096804229864246, 'hand_for_town': 0.02592709169025245, 'street_build_options': 0.15245100032881437, 'village_build_options': 0.001214186459756158, 'cards_earning_power': 0.05875605789823647, 'hand_for_street_missing_one': 0.0025146945753360514, 'hand_for_village_missing_one': 0.0022868318725407866, 'hand_for_town_missing_one': 0.002073570262924233, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.02613168028726115, 'hand_for_village_missing_two': 0.0014588388626225507, 'hand_for_town_missing_two': 0.004294001791959429, 'secondary_options_earning_power': 0.01601568657843535, 'resource_type_weight': np.array([0.2219747 , 0.        , 0.18794585, 0.2010289 , 0.15917066,
       0.2298799 ])}
pref_5 = PlayerPreferences(**d).normalized()
d = {'full_score': 500.0, 'streets': 0.12160336116534015, 'villages': 0.2711269638724776, 'towns': 0.2577147943320559, 'penalty_reference_for_too_many_cards': 7.0, 'cards_in_hand': 0.001437100691403025, 'hand_for_street': 0.030951838944295953, 'hand_for_village': 0.03158418793767723, 'hand_for_town': 0.025957031416196832, 'street_build_options': 0.14587274280830184, 'village_build_options': 0.0012215677171944338, 'cards_earning_power': 0.05719039591421422, 'hand_for_street_missing_one': 0.0025144943233398123, 'hand_for_village_missing_one': 0.002267895476593623, 'hand_for_town_missing_one': 0.0020568839361967636, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.027034223301707877, 'hand_for_village_missing_two': 0.001434706113611424, 'hand_for_town_missing_two': 0.004174152720795571, 'secondary_options_earning_power': 0.015857659328597763, 'resource_type_weight': np.array([0.2189006 , 0.        , 0.18386303, 0.20750293, 0.1602746 ,
       0.22945885])}
pref_6 = PlayerPreferences(**d).normalized()
d = {'full_score': 500.0, 'streets': 0.13431358245398364, 'villages': 0.2711736322260449, 'towns': 0.24224303589061277, 'penalty_reference_for_too_many_cards': 7.0, 'cards_in_hand': 0.0014956692248029777, 'hand_for_street': 0.030258974790008764, 'hand_for_village': 0.02982903080109584, 'hand_for_town': 0.023698683564538636, 'street_build_options': 0.14725941536213402, 'village_build_options': 0.0012815718771795996, 'cards_earning_power': 0.06270410536699057, 'hand_for_street_missing_one': 0.002305936134673843, 'hand_for_village_missing_one': 0.002217398591237843, 'hand_for_town_missing_one': 0.002331404005756716, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.027385663338714925, 'hand_for_village_missing_two': 0.0014200021464839563, 'hand_for_town_missing_two': 0.004258535386292375, 'secondary_options_earning_power': 0.015823358839448614, 'resource_type_weight': np.array([0.24219109, 0.        , 0.1815336 , 0.20734684, 0.14611982,
       0.22280865])}
pref_7 = PlayerPreferences(**d).normalized()
d = {'full_score': 500.0, 'streets': 0.1050360421084255, 'villages': 0.30508049543091376, 'towns': 0.2552276197178769, 'penalty_reference_for_too_many_cards': 7.0, 'cards_in_hand': 0.0016313226955566318, 'hand_for_street': 0.0358453538384792, 'hand_for_village': 0.0317007127920895, 'hand_for_town': 0.02639428755829798, 'street_build_options': 0.12319158604619346, 'village_build_options': 0.0013879698774662522, 'cards_earning_power': 0.06276561696934392, 'hand_for_street_missing_one': 0.002640403000363657, 'hand_for_village_missing_one': 0.0022294508986602825, 'hand_for_town_missing_one': 0.0018672873350750066, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.024124536297141216, 'hand_for_village_missing_two': 0.0014491268181714425, 'hand_for_town_missing_two': 0.0044338069574201494, 'secondary_options_earning_power': 0.014994381658525185, 'resource_type_weight': np.array([0.19394866, 0.        , 0.1742844 , 0.23208966, 0.18669983,
       0.21297746])}
pref_8 = PlayerPreferences(**d).normalized()

d = {'full_score': 500.0, 'streets': 0.16230640903319835, 'villages': 0.2717506229162278, 'towns': 0.23753909881594418, 'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0016416566767053558, 'hand_for_street': 0.026871536336953775, 'hand_for_village': 0.033617591046744014, 'hand_for_town': 0.028509502611393404, 'street_build_options': 0.1366325715199549, 'village_build_options': 0.0014427924734694562, 'cards_earning_power': 0.04557764757789512, 'hand_for_street_missing_one': 0.002745557289286853, 'hand_for_village_missing_one': 0.002174485757513419, 'hand_for_town_missing_one': 0.00179044270534351, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.02587780547450723, 'hand_for_village_missing_two': 0.0015699590716065459, 'hand_for_town_missing_two': 0.004181527351051802, 'secondary_options_earning_power': 0.01577079334220432, 'resource_type_weight': np.array([0.25912856, 0.        , 0.14324592, 0.24272492, 0.14724166,
       0.20765893])}
pref_9 = PlayerPreferences(**d).normalized()
d = {'full_score': 500.0, 'streets': 0.10954150741629823, 'villages': 0.2702064535408607, 'towns': 0.26674838104156107, 'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0015735467063674809, 'hand_for_street': 0.02809932191311624, 'hand_for_village': 0.037254233066562545, 'hand_for_town': 0.020406666908120707, 'street_build_options': 0.15521506355670983, 'village_build_options': 0.001004343271372723, 'cards_earning_power': 0.05466142525373416, 'hand_for_street_missing_one': 0.0021454638345286357, 'hand_for_village_missing_one': 0.0019125389169016129, 'hand_for_town_missing_one': 0.002001541067246348, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.02879779841200556, 'hand_for_village_missing_two': 0.0014265809591500938, 'hand_for_town_missing_two': 0.004181180252502564, 'secondary_options_earning_power': 0.014823953882961524, 'resource_type_weight': np.array([0.29097919, 0.        , 0.1717805 , 0.18755748, 0.15653809,
       0.19314474])}
pref_10 = PlayerPreferences(**d).normalized()
d = {'full_score': 500.0, 'streets': 0.1030502890735972, 'villages': 0.2779812040906803, 'towns': 0.2615685235254844, 'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0016013336066229371, 'hand_for_street': 0.028189994447717644, 'hand_for_village': 0.03827107325718091, 'hand_for_town': 0.020006989985115173, 'street_build_options': 0.15330089974921854, 'village_build_options': 0.0010814496057464681, 'cards_earning_power': 0.05710544048949283, 'hand_for_street_missing_one': 0.0020301883672092158, 'hand_for_village_missing_one': 0.0021816338718114252, 'hand_for_town_missing_one': 0.001888200146127797, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.03097728043393461, 'hand_for_village_missing_two': 0.0014586463058181207, 'hand_for_town_missing_two': 0.0041961211703375285, 'secondary_options_earning_power': 0.015110731873905006, 'resource_type_weight': np.array([0.28933925, 0.        , 0.17325075, 0.17009075, 0.16580734,
       0.20151191])}
pref_11 = PlayerPreferences(**d).normalized()
d = {'full_score': 500.0, 'streets': 0.12841409220651032, 'villages': 0.23200360277843174, 'towns': 0.24735921274586684, 'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0019813092340462742, 'hand_for_street': 0.03499900540674693, 'hand_for_village': 0.04353500958368031, 'hand_for_town': 0.021859239783559622, 'street_build_options': 0.16795331248344703, 'village_build_options': 0.0011699101257809316, 'cards_earning_power': 0.0631354409956274, 'hand_for_street_missing_one': 0.002783503593755327, 'hand_for_village_missing_one': 0.0018610377598767493, 'hand_for_town_missing_one': 0.0026137061801754656, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.03169021054135229, 'hand_for_village_missing_two': 0.001310613509659782, 'hand_for_town_missing_two': 0.0036112154019408743, 'secondary_options_earning_power': 0.013719577669542203, 'resource_type_weight': np.array([0.33884646, 0.        , 0.19709431, 0.18910332, 0.12576471,
       0.14919122])}
pref_12 = PlayerPreferences(**d).normalized()
d = {'full_score': 500.0, 'streets': 0.16230640903319835, 'villages': 0.2717506229162278, 'towns': 0.23753909881594418, 'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0016416566767053558, 'hand_for_street': 0.026871536336953775, 'hand_for_village': 0.033617591046744014, 'hand_for_town': 0.028509502611393404, 'street_build_options': 0.1366325715199549, 'village_build_options': 0.0014427924734694562, 'cards_earning_power': 0.04557764757789512, 'hand_for_street_missing_one': 0.002745557289286853, 'hand_for_village_missing_one': 0.002174485757513419, 'hand_for_town_missing_one': 0.00179044270534351, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.02587780547450723, 'hand_for_village_missing_two': 0.0015699590716065459, 'hand_for_town_missing_two': 0.004181527351051802, 'secondary_options_earning_power': 0.01577079334220432, 'resource_type_weight': np.array([0.25912856, 0.        , 0.14324592, 0.24272492, 0.14724166,
       0.20765893])}
pref_best_so_far = PlayerPreferences(**d).normalized()
d = {'full_score': 500.0, 'streets': 0.18195226103055395, 'villages': 0.23531848548941509, 'towns': 0.2566224359803076, 'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0015358829034074779, 'hand_for_street': 0.02423108890005969, 'hand_for_village': 0.03313457068519678, 'hand_for_town': 0.030577877251824717, 'street_build_options': 0.1270705882505315, 'village_build_options': 0.001345752273741217, 'cards_earning_power': 0.05284921641868388, 'hand_for_street_missing_one': 0.0029245048589268286, 'hand_for_village_missing_one': 0.0021772459628969214, 'hand_for_town_missing_one': 0.0016292948136894525, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.026479365411980843, 'hand_for_village_missing_two': 0.001612619066547681, 'hand_for_town_missing_two': 0.004265160785739681, 'secondary_options_earning_power': 0.01627364991649674, 'resource_type_weight': np.array([0.23897398, 0.        , 0.13003389, 0.28342165, 0.13089159,
       0.2166789 ])}
pref_12 = PlayerPreferences(**d).normalized()
d = {'full_score': 500.0, 'streets': 0.18021819744241216, 'villages': 0.23292077862514943, 'towns': 0.2585507068866446, 'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0015607359981718101, 'hand_for_street': 0.024091966196016307, 'hand_for_village': 0.03339267704979977, 'hand_for_town': 0.030139597335765805, 'street_build_options': 0.129974538590696, 'village_build_options': 0.001336534495402773, 'cards_earning_power': 0.05237539824952986, 'hand_for_street_missing_one': 0.0029170242121083437, 'hand_for_village_missing_one': 0.0022119965802210374, 'hand_for_town_missing_one': 0.001608828341530381, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.026304409609836148, 'hand_for_village_missing_two': 0.0016309842271974779, 'hand_for_town_missing_two': 0.004371032675720943, 'secondary_options_earning_power': 0.016394593483797147, 'resource_type_weight': np.array([0.23698116, 0.        , 0.12862897, 0.28597763, 0.13045122,
       0.21796103])}
pref_13 = PlayerPreferences(**d).normalized()
d = {'full_score': 500.0, 'streets': 0.1831915540882006, 'villages': 0.2256727441134013, 'towns': 0.26346256081566616, 'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0015384426069901523, 'hand_for_street': 0.023145564113870556, 'hand_for_village': 0.03363987354186871, 'hand_for_town': 0.030319509365970738, 'street_build_options': 0.12902564645659567, 'village_build_options': 0.0013672000432326129, 'cards_earning_power': 0.053675480314995985, 'hand_for_street_missing_one': 0.0028912126469644982, 'hand_for_village_missing_one': 0.0022098843970663816, 'hand_for_town_missing_one': 0.0015746694731949991, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.025892966563584124, 'hand_for_village_missing_two': 0.0015798737968834864, 'hand_for_town_missing_two': 0.004243026941238099, 'secondary_options_earning_power': 0.016569790720276017, 'resource_type_weight': np.array([0.23887194, 0.        , 0.1291717 , 0.28629221, 0.12811895,
       0.21754521])}
pref_14 = PlayerPreferences(**d).normalized()

d = {'full_score': 500.0, 'streets': 0.13642071663815078, 'villages': 0.2901284337947034, 'towns': 0.23319439131827813, 'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0017843670492005595, 'hand_for_street': 0.023453701872963088, 'hand_for_village': 0.039335324566285135, 'hand_for_town': 0.030178451358922973, 'street_build_options': 0.12958860336451064, 'village_build_options': 0.0014181106079347288, 'cards_earning_power': 0.05510079192392932, 'hand_for_street_missing_one': 0.0028082275705629163, 'hand_for_village_missing_one': 0.0023981546756130846, 'hand_for_town_missing_one': 0.0016719670770469603, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.028040183890494286, 'hand_for_village_missing_two': 0.0017705756113798362, 'hand_for_town_missing_two': 0.0039836384588558356, 'secondary_options_earning_power': 0.018724360221168378, 'resource_type_weight': np.array([0.24684187, 0.        , 0.13813439, 0.26095626, 0.17211745,
       0.18195002])}
pref_GE_1 = PlayerPreferences(**d).normalized()
d = {'full_score': 500.0, 'streets': 0.13546340697160333, 'villages': 0.28955252470914283, 'towns': 0.23679377607281915, 'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0017065066878841361, 'hand_for_street': 0.022729544398786144, 'hand_for_village': 0.03973547848958623, 'hand_for_town': 0.030041854407907888, 'street_build_options': 0.1300545280186492, 'village_build_options': 0.0014243720760441169, 'cards_earning_power': 0.05435934308893406, 'hand_for_street_missing_one': 0.002802089297772036, 'hand_for_village_missing_one': 0.0024008537078616158, 'hand_for_town_missing_one': 0.0016768896528647674, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.02724319859328979, 'hand_for_village_missing_two': 0.0017718523042243688, 'hand_for_town_missing_two': 0.003889626311933226, 'secondary_options_earning_power': 0.018354155210697165, 'resource_type_weight': np.array([0.24683585, 0.        , 0.14041006, 0.26005674, 0.17206154,
       0.1806358 ])}
pref_GE_2 = PlayerPreferences(**d).normalized()
d = {'full_score': 500.0, 'streets': 0.13437135208676795, 'villages': 0.28218015944335556, 'towns': 0.2406037586237538, 'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0017602003522712096, 'hand_for_street': 0.02367520132612697, 'hand_for_village': 0.04075581506730396, 'hand_for_town': 0.029123891142644284, 'street_build_options': 0.13106362450693776, 'village_build_options': 0.0013883637437913734, 'cards_earning_power': 0.056508109377654846, 'hand_for_street_missing_one': 0.00280854584974505, 'hand_for_village_missing_one': 0.002376057012535915, 'hand_for_town_missing_one': 0.0016274115071242927, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.027604089074275816, 'hand_for_village_missing_two': 0.0017001707672506146, 'hand_for_town_missing_two': 0.003938796225774805, 'secondary_options_earning_power': 0.018514453892685814, 'resource_type_weight': np.array([0.24764016, 0.        , 0.13886453, 0.27137876, 0.1676857 ,
       0.17443084])}
pref_GE_3 = PlayerPreferences(**d).normalized()
d = {'full_score': 500.0, 'streets': 0.14602672852112267, 'villages': 0.2884670928798847, 'towns': 0.2310985927424778, 'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0017825680504437885, 'hand_for_street': 0.021387757943406, 'hand_for_village': 0.037621012787202236, 'hand_for_town': 0.03149600745933846, 'street_build_options': 0.1324450435241632, 'village_build_options': 0.0015214894984841535, 'cards_earning_power': 0.04984318203522712, 'hand_for_street_missing_one': 0.002993907498006098, 'hand_for_village_missing_one': 0.002331602596337053, 'hand_for_town_missing_one': 0.001739285135532411, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.028058460306861967, 'hand_for_village_missing_two': 0.0017633224261007327, 'hand_for_town_missing_two': 0.004214386431963903, 'secondary_options_earning_power': 0.017209560163447733, 'resource_type_weight': np.array([0.24776852, 0.        , 0.14737394, 0.26302598, 0.16314134,
       0.17869022])}
pref_GE_4 = PlayerPreferences(**d).normalized()

d={'full_score': 500.0, 'streets': 0.1615594227747923, 'villages': 0.2961485945875825, 'towns': 0.24153440764240208, 'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0013711796157184063, 'hand_for_street': 0.023639163335292715, 'hand_for_village': 0.03128547202248002, 'hand_for_town': 0.025491342005668043, 'street_build_options': 0.11401320207729074, 'village_build_options': 0.0014075613236915708, 'cards_earning_power': 0.05033539418219382, 'hand_for_street_missing_one': 0.002443204306152173, 'hand_for_village_missing_one': 0.002042238708140502, 'hand_for_town_missing_one': 0.0017278376628582053, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.026728644648182663, 'hand_for_village_missing_two': 0.0014441706982328611, 'hand_for_town_missing_two': 0.0036821948583871038, 'secondary_options_earning_power': 0.015145969550934273, 'resource_type_weight': np.array([0.27126462, 0.        , 0.13846242, 0.25696639, 0.15221156,
       0.18109501])}
pref_GE_5 = PlayerPreferences(**d).normalized()
d={'full_score': 500.0, 'streets': 0.16082165423744554, 'villages': 0.2969455006422883, 'towns': 0.24182636274474084, 'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0013760033525157503, 'hand_for_street': 0.023567214895835527, 'hand_for_village': 0.03131130427939963, 'hand_for_town': 0.025509922605764117, 'street_build_options': 0.11377529271055659, 'village_build_options': 0.001412191190089865, 'cards_earning_power': 0.050382892959078995, 'hand_for_street_missing_one': 0.002447298527055522, 'hand_for_village_missing_one': 0.0020378587598259106, 'hand_for_town_missing_one': 0.0017242248440004306, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.026603552412143253, 'hand_for_village_missing_two': 0.0014524782657998946, 'hand_for_town_missing_two': 0.003666544480475352, 'secondary_options_earning_power': 0.01513970309298444, 'resource_type_weight': np.array([0.27171115, 0.        , 0.1391797 , 0.25674677, 0.15160797,
       0.18075441])}
pref_GE_6 = PlayerPreferences(**d).normalized()
d={'full_score': 500.0, 'streets': 0.16144506065322062, 'villages': 0.29617556915247584, 'towns': 0.2412229816605218, 'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0013870037041360706, 'hand_for_street': 0.02360704482693601, 'hand_for_village': 0.031622409347721335, 'hand_for_town': 0.025680107950018037, 'street_build_options': 0.1140313988248067, 'village_build_options': 0.0014120658319310164, 'cards_earning_power': 0.05035103761542655, 'hand_for_street_missing_one': 0.002439813144932014, 'hand_for_village_missing_one': 0.002053510639914122, 'hand_for_town_missing_one': 0.0017226849900645973, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.0264864035021441, 'hand_for_village_missing_two': 0.0014709580833413723, 'hand_for_town_missing_two': 0.003648477370613058, 'secondary_options_earning_power': 0.015243472701796809, 'resource_type_weight': np.array([0.27229092, 0.        , 0.13957448, 0.25620405, 0.15156548,
       0.18036507])}
pref_GE_7 = PlayerPreferences(**d).normalized()
d={'full_score': 500.0, 'streets': 0.16038378927223754, 'villages': 0.29816309038060196, 'towns': 0.24332782148487025, 'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0013667451429243776, 'hand_for_street': 0.023348042208727677, 'hand_for_village': 0.031202429737339565, 'hand_for_town': 0.02524180105543993, 'street_build_options': 0.11281999073658096, 'village_build_options': 0.001387010178369213, 'cards_earning_power': 0.049622424212071414, 'hand_for_street_missing_one': 0.002408573907716494, 'hand_for_village_missing_one': 0.0019996085653786458, 'hand_for_town_missing_one': 0.0017357117261289944, 'secondary_village_build_options': 0.0, 'direct_options_earning_power': 0.026794709352327424, 'hand_for_village_missing_two': 0.0014523311961166787, 'hand_for_town_missing_two': 0.003605656555547883, 'secondary_options_earning_power': 0.015140264287621009, 'resource_type_weight': np.array([0.26959237, 0.        , 0.13741446, 0.25547097, 0.15356377,
       0.18395843])}
pref_GE_8 = PlayerPreferences(**d).normalized()

d = {'full_score': 1000.0, 'streets': 10, 'villages': 20, 'towns': 20,
    'penalty_reference_for_too_many_cards': 7.0, 'cards_in_hand': 1, 'hand_for_street':0, 'hand_for_village': 0,
    'hand_for_town': 0, 'street_build_options': 0, 'village_build_options': 0, 'cards_earning_power': 0,
        'hand_for_street_missing_one': 0, 'hand_for_village_missing_one': 0, 'hand_for_town_missing_one': 0,
        'secondary_village_build_options': 0, 'direct_options_earning_power': 2, 'hand_for_village_missing_two': 0,
    'hand_for_town_missing_two': 0, 'secondary_options_earning_power': 0,
    'resource_type_weight': np.array([0, 0 , 1, 2, 1, 2])}
mediocre_1 = PlayerPreferences(**d).normalized()

d = {'full_score': 500.0, 'streets': 0, 'villages': 0, 'towns': 0,
    'penalty_reference_for_too_many_cards': 7.0, 'cards_in_hand': 0, 'hand_for_street':0, 'hand_for_village': 0,
    'hand_for_town': 0, 'street_build_options': 0.1, 'village_build_options': 0, 'cards_earning_power': 0,
        'hand_for_street_missing_one': 0, 'hand_for_village_missing_one': 0, 'hand_for_town_missing_one': 0,
        'secondary_village_build_options': 0, 'direct_options_earning_power': 0, 'hand_for_village_missing_two': 0,
    'hand_for_town_missing_two': 0, 'secondary_options_earning_power':0,
    'resource_type_weight': np.array([0.1, 0. , 0.3, 0.3, 0.2, 0.1])}
mediocre_2 = PlayerPreferences(**d).normalized()
 
d = {'full_score': 1000.0, 'streets': 10, 'villages': 20, 'towns': 20,
    'penalty_reference_for_too_many_cards': 7.0, 'cards_in_hand': 0.1, 'hand_for_street':2, 'hand_for_village': 2,
    'hand_for_town': 2, 'street_build_options': 0, 'village_build_options': 0.1, 'cards_earning_power': 5,
        'hand_for_street_missing_one': 0.2, 'hand_for_village_missing_one': 0.2, 'hand_for_town_missing_one': 0.2,
        'secondary_village_build_options': 0, 'direct_options_earning_power': 2, 'hand_for_village_missing_two': 0.1,
    'hand_for_town_missing_two': 0.1, 'secondary_options_earning_power': 1,
    'resource_type_weight': np.array([2, 0. , 1, 1, 1, 2])}
strong_1 = PlayerPreferences(**d).normalized()

d = {'full_score': 500.0, 'streets': 10, 'villages': 10, 'towns': 20,
    'penalty_reference_for_too_many_cards': 7.0, 'cards_in_hand': 0.1, 'hand_for_street':2, 'hand_for_village': 2,
    'hand_for_town': 2, 'street_build_options': 0.1, 'village_build_options': 0.1, 'cards_earning_power': 0.2,
        'hand_for_street_missing_one': 0, 'hand_for_village_missing_one': 0, 'hand_for_town_missing_one': 0,
        'secondary_village_build_options': 0, 'direct_options_earning_power': 0.1, 'hand_for_village_missing_two': 0.1,
    'hand_for_town_missing_two': 1, 'secondary_options_earning_power':1,
    'resource_type_weight': np.array([0.1, 0. , 0.3, 0.3, 0.2, 0.1])}
strong_2 = PlayerPreferences(**d).normalized()

d={'full_score': 500.0, 'streets': 0.1615594227747923, 'villages': 0.2961485945875825, 'towns': 0.24153440764240208, 
   'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0013711796157184063, 
   'hand_for_street': 0.023639163335292715, 'hand_for_village': 0.03128547202248002, 
   'hand_for_town': 0.025491342005668043, 'street_build_options': 0.11401320207729074, 
   'village_build_options': 0.0014075613236915708, 'cards_earning_power': 0.05033539418219382, 
   'hand_for_street_missing_one': 0.002443204306152173, 'hand_for_village_missing_one': 0.002042238708140502, 
   'hand_for_town_missing_one': 0.0017278376628582053, 'secondary_village_build_options': 0.0, 
   'direct_options_earning_power': 0.026728644648182663, 'hand_for_village_missing_two': 0.0014441706982328611, 
   'hand_for_town_missing_two': 0.0036821948583871038, 'secondary_options_earning_power': 0.015145969550934273, 
   'resource_type_weight': np.array([0.27126462, 0.        , 0.13846242, 0.25696639, 0.15221156,
       0.18109501])}
optimized_1 = PlayerPreferences(**d).normalized()

d = {'full_score': 500.0, 'streets': 0.16230640903319835, 'villages': 0.2717506229162278, 'towns': 0.23753909881594418, 
     'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0016416566767053558, 
     'hand_for_street': 0.026871536336953775, 'hand_for_village': 0.033617591046744014, 
     'hand_for_town': 0.028509502611393404, 'street_build_options': 0.1366325715199549, 
     'village_build_options': 0.0014427924734694562, 'cards_earning_power': 0.04557764757789512, 
     'hand_for_street_missing_one': 0.002745557289286853, 'hand_for_village_missing_one': 0.002174485757513419,
     'hand_for_town_missing_one': 0.00179044270534351, 'secondary_village_build_options': 0.0, 
     'direct_options_earning_power': 0.02587780547450723, 'hand_for_village_missing_two': 0.0015699590716065459, 
     'hand_for_town_missing_two': 0.004181527351051802, 'secondary_options_earning_power': 0.01577079334220432, 
     'resource_type_weight': np.array([0.25912856, 0.        , 0.14324592, 0.24272492, 0.14724166,
       0.20765893])}
optimized_2 = PlayerPreferences(**d).normalized()