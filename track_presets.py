# track_presets.py
# 2026 F1 Calendar - Race configuration presets
# Lap times and tire parameters are approximate based on historical data
# and 2026 car estimates (~2s slower than 2025 per regulation changes)

TRACK_PRESETS = {
    "Custom": None,

    # Round 1 - March 8
    "Australia (Melbourne)": {
        'base_lap_time': 83, 'total_laps': 58, 'pit_loss': 23,
        'sc_chance_pct': 60,  # Albert Park is SC-prone
        'base_traffic_prob': 0.35, 'spread_rate': 15, 'traffic_time_loss': 0.5,
        'sc_pit_loss': 11, 'sc_base_lap_time': 92,
        'soft_offset': -1.0, 'soft_deg': 0.09, 'soft_max': 14,
        'med_offset': 0.0,  'med_deg': 0.05, 'med_max': 32,
        'hard_offset': 1.3, 'hard_deg': 0.02, 'hard_max': 50,
    },

    # Round 2 - March 15
    "China (Shanghai)": {
        'base_lap_time': 95, 'total_laps': 56, 'pit_loss': 24,
        'sc_chance_pct': 40,
        'base_traffic_prob': 0.35, 'spread_rate': 18, 'traffic_time_loss': 0.4,
        'sc_pit_loss': 11, 'sc_base_lap_time': 105,
        'soft_offset': -1.0, 'soft_deg': 0.10, 'soft_max': 12,
        'med_offset': 0.0,  'med_deg': 0.06, 'med_max': 30,
        'hard_offset': 1.2, 'hard_deg': 0.02, 'hard_max': 50,
    },

    # Round 3 - March 29
    "Japan (Suzuka)": {
        'base_lap_time': 93, 'total_laps': 53, 'pit_loss': 22,
        'sc_chance_pct': 35,
        'base_traffic_prob': 0.3, 'spread_rate': 20, 'traffic_time_loss': 0.4,
        'sc_pit_loss': 11, 'sc_base_lap_time': 103,
        'soft_offset': -1.0, 'soft_deg': 0.07, 'soft_max': 16,
        'med_offset': 0.0,  'med_deg': 0.04, 'med_max': 36,
        'hard_offset': 1.2, 'hard_deg': 0.02, 'hard_max': 55,
    },

    # Round 4 - April 12 (may be cancelled)
    "Bahrain (Sakhir)": {
        'base_lap_time': 95, 'total_laps': 57, 'pit_loss': 22,
        'sc_chance_pct': 30,
        'base_traffic_prob': 0.3, 'spread_rate': 18, 'traffic_time_loss': 0.4,
        'sc_pit_loss': 10, 'sc_base_lap_time': 105,
        'soft_offset': -1.0, 'soft_deg': 0.12, 'soft_max': 10,  # very high deg
        'med_offset': 0.0,  'med_deg': 0.07, 'med_max': 25,
        'hard_offset': 1.0, 'hard_deg': 0.03, 'hard_max': 45,
    },

    # Round 5 - April 19 (may be cancelled)
    "Saudi Arabia (Jeddah)": {
        'base_lap_time': 90, 'total_laps': 50, 'pit_loss': 24,
        'sc_chance_pct': 55,  # street circuit, SC-prone
        'base_traffic_prob': 0.4, 'spread_rate': 14, 'traffic_time_loss': 0.6,
        'sc_pit_loss': 12, 'sc_base_lap_time': 100,
        'soft_offset': -1.0, 'soft_deg': 0.08, 'soft_max': 15,
        'med_offset': 0.0,  'med_deg': 0.05, 'med_max': 32,
        'hard_offset': 1.3, 'hard_deg': 0.02, 'hard_max': 50,
    },

    # Round 6 - May 3
    "Miami": {
        'base_lap_time': 92, 'total_laps': 57, 'pit_loss': 23,
        'sc_chance_pct': 50,
        'base_traffic_prob': 0.4, 'spread_rate': 15, 'traffic_time_loss': 0.5,
        'sc_pit_loss': 11, 'sc_base_lap_time': 102,
        'soft_offset': -1.0, 'soft_deg': 0.09, 'soft_max': 13,
        'med_offset': 0.0,  'med_deg': 0.06, 'med_max': 30,
        'hard_offset': 1.2, 'hard_deg': 0.02, 'hard_max': 50,
    },

    # Round 7 - May 24
    "Canada (Montreal)": {
        'base_lap_time': 77, 'total_laps': 70, 'pit_loss': 23,
        'sc_chance_pct': 60,  # historically very SC-prone
        'base_traffic_prob': 0.4, 'spread_rate': 14, 'traffic_time_loss': 0.5,
        'sc_pit_loss': 11, 'sc_base_lap_time': 87,
        'soft_offset': -1.0, 'soft_deg': 0.08, 'soft_max': 15,
        'med_offset': 0.0,  'med_deg': 0.05, 'med_max': 33,
        'hard_offset': 1.2, 'hard_deg': 0.02, 'hard_max': 52,
    },

    # Round 8 - June 7
    "Monaco": {
        'base_lap_time': 76, 'total_laps': 78, 'pit_loss': 28,  # long pit lane
        'sc_chance_pct': 65,  # very SC-prone street circuit
        'base_traffic_prob': 0.7, 'spread_rate': 8, 'traffic_time_loss': 1.5,  # heavy traffic
        'sc_pit_loss': 14, 'sc_base_lap_time': 85,
        'soft_offset': -1.0, 'soft_deg': 0.05, 'soft_max': 25,  # low deg, smooth surface
        'med_offset': 0.0,  'med_deg': 0.03, 'med_max': 45,
        'hard_offset': 1.0, 'hard_deg': 0.01, 'hard_max': 70,
    },

    # Round 9 - June 14
    "Barcelona-Catalunya": {
        'base_lap_time': 83, 'total_laps': 66, 'pit_loss': 22,
        'sc_chance_pct': 30,
        'base_traffic_prob': 0.3, 'spread_rate': 18, 'traffic_time_loss': 0.4,
        'sc_pit_loss': 10, 'sc_base_lap_time': 93,
        'soft_offset': -1.0, 'soft_deg': 0.09, 'soft_max': 14,
        'med_offset': 0.0,  'med_deg': 0.06, 'med_max': 30,
        'hard_offset': 1.2, 'hard_deg': 0.02, 'hard_max': 50,
    },

    # Round 10 - June 28
    "Austria (Spielberg)": {
        'base_lap_time': 68, 'total_laps': 71, 'pit_loss': 21,
        'sc_chance_pct': 40,
        'base_traffic_prob': 0.35, 'spread_rate': 16, 'traffic_time_loss': 0.4,
        'sc_pit_loss': 10, 'sc_base_lap_time': 78,
        'soft_offset': -1.0, 'soft_deg': 0.09, 'soft_max': 14,
        'med_offset': 0.0,  'med_deg': 0.06, 'med_max': 30,
        'hard_offset': 1.2, 'hard_deg': 0.02, 'hard_max': 50,
    },

    # Round 11 - July 5
    "Great Britain (Silverstone)": {
        'base_lap_time': 92, 'total_laps': 52, 'pit_loss': 23,
        'sc_chance_pct': 45,
        'base_traffic_prob': 0.3, 'spread_rate': 18, 'traffic_time_loss': 0.4,
        'sc_pit_loss': 11, 'sc_base_lap_time': 102,
        'soft_offset': -1.0, 'soft_deg': 0.08, 'soft_max': 16,
        'med_offset': 0.0,  'med_deg': 0.05, 'med_max': 34,
        'hard_offset': 1.2, 'hard_deg': 0.02, 'hard_max': 52,
    },

    # Round 12 - July 19
    "Belgium (Spa)": {
        'base_lap_time': 107, 'total_laps': 44, 'pit_loss': 23,
        'sc_chance_pct': 50,
        'base_traffic_prob': 0.3, 'spread_rate': 20, 'traffic_time_loss': 0.4,
        'sc_pit_loss': 11, 'sc_base_lap_time': 118,
        'soft_offset': -1.0, 'soft_deg': 0.07, 'soft_max': 18,
        'med_offset': 0.0,  'med_deg': 0.04, 'med_max': 36,
        'hard_offset': 1.0, 'hard_deg': 0.02, 'hard_max': 55,
    },

    # Round 13 - July 26
    "Hungary (Budapest)": {
        'base_lap_time': 81, 'total_laps': 70, 'pit_loss': 22,
        'sc_chance_pct': 30,
        'base_traffic_prob': 0.4, 'spread_rate': 14, 'traffic_time_loss': 0.6,  # overtaking hard
        'sc_pit_loss': 10, 'sc_base_lap_time': 91,
        'soft_offset': -1.0, 'soft_deg': 0.08, 'soft_max': 15,
        'med_offset': 0.0,  'med_deg': 0.05, 'med_max': 32,
        'hard_offset': 1.2, 'hard_deg': 0.02, 'hard_max': 50,
    },

    # Round 14 - August (TBC - Dutch GP last year before removal)
    "Netherlands (Zandvoort)": {
        'base_lap_time': 73, 'total_laps': 72, 'pit_loss': 22,
        'sc_chance_pct': 40,
        'base_traffic_prob': 0.45, 'spread_rate': 12, 'traffic_time_loss': 0.7,
        'sc_pit_loss': 10, 'sc_base_lap_time': 83,
        'soft_offset': -1.0, 'soft_deg': 0.10, 'soft_max': 12,
        'med_offset': 0.0,  'med_deg': 0.06, 'med_max': 28,
        'hard_offset': 1.2, 'hard_deg': 0.03, 'hard_max': 45,
    },

    # Round 15 - September 13
    "Spain (Madrid)": {
        'base_lap_time': 90, 'total_laps': 55, 'pit_loss': 25,  # new street circuit
        'sc_chance_pct': 55,
        'base_traffic_prob': 0.45, 'spread_rate': 13, 'traffic_time_loss': 0.6,
        'sc_pit_loss': 12, 'sc_base_lap_time': 100,
        'soft_offset': -1.0, 'soft_deg': 0.09, 'soft_max': 14,
        'med_offset': 0.0,  'med_deg': 0.06, 'med_max': 30,
        'hard_offset': 1.2, 'hard_deg': 0.02, 'hard_max': 50,
    },

    # Round 16 - September 26 (Saturday race)
    "Azerbaijan (Baku)": {
        'base_lap_time': 106, 'total_laps': 51, 'pit_loss': 25,
        'sc_chance_pct': 65,  # very SC-prone street circuit
        'base_traffic_prob': 0.4, 'spread_rate': 14, 'traffic_time_loss': 0.6,
        'sc_pit_loss': 12, 'sc_base_lap_time': 116,
        'soft_offset': -1.0, 'soft_deg': 0.07, 'soft_max': 20,
        'med_offset': 0.0,  'med_deg': 0.04, 'med_max': 38,
        'hard_offset': 1.0, 'hard_deg': 0.02, 'hard_max': 58,
    },

    # Round 17 - October 11
    "Singapore": {
        'base_lap_time': 101, 'total_laps': 62, 'pit_loss': 27,
        'sc_chance_pct': 70,  # most SC-prone race on calendar
        'base_traffic_prob': 0.5, 'spread_rate': 10, 'traffic_time_loss': 0.8,
        'sc_pit_loss': 13, 'sc_base_lap_time': 111,
        'soft_offset': -1.0, 'soft_deg': 0.06, 'soft_max': 22,
        'med_offset': 0.0,  'med_deg': 0.04, 'med_max': 40,
        'hard_offset': 1.0, 'hard_deg': 0.02, 'hard_max': 60,
    },

    # Round 18 - October 25
    "USA (Austin)": {
        'base_lap_time': 98, 'total_laps': 56, 'pit_loss': 23,
        'sc_chance_pct': 45,
        'base_traffic_prob': 0.35, 'spread_rate': 16, 'traffic_time_loss': 0.5,
        'sc_pit_loss': 11, 'sc_base_lap_time': 108,
        'soft_offset': -1.0, 'soft_deg': 0.09, 'soft_max': 14,
        'med_offset': 0.0,  'med_deg': 0.06, 'med_max': 30,
        'hard_offset': 1.2, 'hard_deg': 0.02, 'hard_max': 50,
    },

    # Round 19 - November 1
    "Mexico City": {
        'base_lap_time': 81, 'total_laps': 71, 'pit_loss': 23,
        'sc_chance_pct': 35,
        'base_traffic_prob': 0.35, 'spread_rate': 16, 'traffic_time_loss': 0.5,
        'sc_pit_loss': 11, 'sc_base_lap_time': 91,
        'soft_offset': -0.8, 'soft_deg': 0.07, 'soft_max': 18,  # high altitude = less deg
        'med_offset': 0.0,  'med_deg': 0.04, 'med_max': 38,
        'hard_offset': 1.0, 'hard_deg': 0.02, 'hard_max': 58,
    },

    # Round 20 - November 8
    "Brazil (São Paulo)": {
        'base_lap_time': 73, 'total_laps': 71, 'pit_loss': 22,
        'sc_chance_pct': 60,
        'base_traffic_prob': 0.4, 'spread_rate': 15, 'traffic_time_loss': 0.5,
        'sc_pit_loss': 11, 'sc_base_lap_time': 83,
        'soft_offset': -1.0, 'soft_deg': 0.09, 'soft_max': 14,
        'med_offset': 0.0,  'med_deg': 0.06, 'med_max': 30,
        'hard_offset': 1.2, 'hard_deg': 0.02, 'hard_max': 50,
    },

    # Round 21 - November 21 (Saturday race)
    "Las Vegas": {
        'base_lap_time': 96, 'total_laps': 50, 'pit_loss': 24,
        'sc_chance_pct': 50,
        'base_traffic_prob': 0.35, 'spread_rate': 16, 'traffic_time_loss': 0.5,
        'sc_pit_loss': 12, 'sc_base_lap_time': 106,
        'soft_offset': -1.0, 'soft_deg': 0.06, 'soft_max': 22,  # low deg, smooth surface
        'med_offset': 0.0,  'med_deg': 0.04, 'med_max': 40,
        'hard_offset': 1.0, 'hard_deg': 0.02, 'hard_max': 60,
    },

    # Round 22 - November 29
    "Qatar (Lusail)": {
        'base_lap_time': 85, 'total_laps': 57, 'pit_loss': 22,
        'sc_chance_pct': 35,
        'base_traffic_prob': 0.3, 'spread_rate': 18, 'traffic_time_loss': 0.4,
        'sc_pit_loss': 10, 'sc_base_lap_time': 95,
        'soft_offset': -1.0, 'soft_deg': 0.11, 'soft_max': 11,  # very high deg
        'med_offset': 0.0,  'med_deg': 0.07, 'med_max': 26,
        'hard_offset': 1.0, 'hard_deg': 0.03, 'hard_max': 44,
    },

    # Round 23 - December 6
    "Abu Dhabi (Yas Marina)": {
        'base_lap_time': 88, 'total_laps': 58, 'pit_loss': 23,
        'sc_chance_pct': 30,
        'base_traffic_prob': 0.3, 'spread_rate': 18, 'traffic_time_loss': 0.4,
        'sc_pit_loss': 11, 'sc_base_lap_time': 98,
        'soft_offset': -1.0, 'soft_deg': 0.08, 'soft_max': 16,
        'med_offset': 0.0,  'med_deg': 0.05, 'med_max': 33,
        'hard_offset': 1.2, 'hard_deg': 0.02, 'hard_max': 52,
    },
}