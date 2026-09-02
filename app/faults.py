"""Reference content for the fault detail panel, keyed by the LOCATION_LABELS slugs.

Two kinds of content live here and they have different standing:

  `measured` and `reliability` are DERIVED FROM THIS PROJECT'S OWN CODE -- the feature
  definitions in envelope_dataset_mcc5.py and the leave-one-condition-out, file-level
  validation recorded in the presence_mcc5.py docstring. Both are checkable against source.

  `causes`, `if_ignored`, `action` and `prevention` are PLACEHOLDER maintenance guidance.
  Review them against course material before the presentation -- confidently wrong repair
  advice is worse than none, and it is the one part of this app a reviewer in the field
  can fact-check on sight.

The system reads three phase currents and nothing else. Every feature is computed from the
envelope (Hilbert magnitude) spectrum of those currents, so a fault is visible here only if
it modulates the current. A purely mechanical fault is invisible however severe.

Reliability figures are precision/recall/F1 at the tuned presence threshold of 0.3. That
threshold was lowered from 0.5 deliberately, to catch faults rather than avoid false
alarms, so weak precision on some locations is a chosen trade rather than an accident.
"""

FAULTS = {
    "bearing_outer": {
        "summary": "Damage to the fixed outer race of the rolling-element bearing.",
        "measured": "Envelope-spectrum magnitude at BPFO = 3.585x shaft frequency "
                    "(+/-2 Hz), per phase. The ratio is the SKF 6205 2Z-C3 value published "
                    "with the dataset, cross-checked against the bearing geometry formula.",
        "reliability": {
            "precision": 0.38, "recall": 0.71, "f1": 0.50,
            "note": "Catches most real outer-race faults, but roughly three in five flags "
                    "are false alarms. Read it as a prompt to inspect, not a diagnosis.",
        },
        "causes": ["Contamination or moisture ingress",
                   "Lubricant loss or degradation",
                   "Shaft currents pitting the race on inverter-fed motors",
                   "Misalignment loading the race outside its contact zone"],
        "if_ignored": "Spalling spreads along the race and the bearing eventually seizes, "
                      "which can score the shaft and turn a bearing change into a rewind.",
        "action": "Inspect at the next maintenance window; check alignment and lubricant "
                  "condition while it is open.",
        "prevention": ["Regreasing on interval with the correct grade and quantity",
                       "Shaft grounding rings on inverter-fed motors",
                       "Sealing appropriate to the environment"],
    },
    "bearing_inner": {
        "summary": "Damage to the rotating inner race of the rolling-element bearing.",
        "measured": "Envelope-spectrum magnitude at BPFI = 5.415x shaft frequency "
                    "(+/-2 Hz), per phase, from the same SKF 6205 2Z-C3 ratios.",
        "reliability": {
            "precision": 0.50, "recall": 0.64, "f1": 0.56,
            "note": "The most dependable of the three bearing locations, and still only "
                    "about half of flags are genuine. Confirm before ordering parts.",
        },
        "causes": ["Lubricant loss or degradation",
                   "Overload, or too tight a fit on the shaft",
                   "Shaft currents on inverter-fed motors",
                   "Damage at installation from pressing on the wrong race"],
        "if_ignored": "Progresses faster than outer-race damage because the defect sits in "
                      "the loaded rotating path. Ends in seizure and likely shaft damage.",
        "action": "Inspect and schedule replacement. Check the fit and mounting method if "
                  "this bearing was recently changed.",
        "prevention": ["Press only on the fitted race when mounting",
                       "Regreasing on interval",
                       "Verify shaft and housing tolerances at every change"],
    },
    "bearing_ball": {
        "summary": "Damage to the rolling elements themselves, or the cage retaining them.",
        "measured": "Envelope-spectrum magnitude at BSF = 2.357x shaft frequency "
                    "(+/-2 Hz), per phase.",
        "reliability": {
            "precision": 0.16, "recall": 0.42, "f1": 0.23,
            "note": "Weak. Most flags are false and most real faults are missed -- a "
                    "rolling-element defect couples into the current far less strongly "
                    "than a race defect does. Corroborate before acting on this alone.",
        },
        "causes": ["Contamination trapped between element and race",
                   "Lubricant breakdown at elevated temperature",
                   "Overload or shock loading",
                   "Cage wear letting elements skew"],
        "if_ignored": "Cage failure lets elements bunch and jam the bearing suddenly, with "
                      "less warning than race damage gives.",
        "action": "Inspect if other evidence agrees. Sample the lubricant for contamination "
                  "if the cause is not obvious.",
        "prevention": ["Clean handling and storage of replacement bearings",
                       "Lubricant grade matched to operating temperature",
                       "Avoid shock loading during coupling"],
    },
    "rotor_bar": {
        "summary": "One or more broken or cracked bars in the squirrel-cage rotor.",
        "measured": "Envelope magnitude at the twice-slip sideband, 2ksf_e, computed per "
                    "phase from the detected supply frequency and an estimated slip. Slip "
                    "comes from nominal RPM rather than direct measurement -- the keyphase "
                    "channel was too noisy to use -- so it is an approximation.",
        "reliability": {
            "precision": 0.54, "recall": 0.83, "f1": 0.65,
            "note": "Strong recall: real broken bars are rarely missed. About half of flags "
                    "are false, and confidence drops further at light load, where slip is "
                    "small and the sideband sits close to the fundamental.",
        },
        "causes": ["Thermal stress from repeated direct-on-line starting",
                   "Casting voids or porosity in the rotor cage",
                   "Cyclic mechanical stress from a pulsating load",
                   "Prolonged operation at high slip"],
        "if_ignored": "Current redistributes to neighbouring bars, which overheat and fail "
                      "in turn. Lifted fragments can strike the stator.",
        "action": "Confirm under load before acting -- this diagnosis is weakest at light "
                  "load. Then plan a rotor inspection.",
        "prevention": ["Soft starters or VFDs to limit starting current",
                       "Limit consecutive starts per hour",
                       "Size the motor to the actual duty cycle"],
    },
    "static_eccentricity": {
        "summary": "The rotor sits off-centre in the stator bore, with the narrowest air "
                   "gap fixed in one position.",
        "measured": "Envelope-spectrum peak at twice shaft rotation frequency. The rigorous "
                    "rotor-slot-harmonic method needs a slot count the dataset does not "
                    "publish, so this uses the generic 2x signature, checked empirically to "
                    "separate static eccentricity from healthy and from mechanical faults.",
        "reliability": {
            "precision": 0.77, "recall": 0.83, "f1": 0.80,
            "note": "The second most reliable location in the system. Supply voltage "
                    "unbalance also inflates this peak, so the two are separated by later "
                    "gating rather than by this feature on its own.",
        },
        "causes": ["Misaligned or worn end shields",
                   "Incorrect assembly after a repair",
                   "Soft foot -- an uneven mounting surface distorting the frame",
                   "Bearing housing wear"],
        "if_ignored": "A constant one-directional magnetic pull accelerates bearing wear "
                      "and can end in rotor-to-stator rub.",
        "action": "Check mounting and foot flatness; verify the air gap at four points "
                  "around the bore if the motor can be opened.",
        "prevention": ["Check for soft foot after every remount",
                       "Alignment on installation and after coupling work",
                       "Machined shims rather than improvised packing"],
    },
    "dynamic_eccentricity": {
        "summary": "The narrowest point of the air gap rotates with the shaft -- the rotor "
                   "is not turning about its own centre.",
        "measured": "No dedicated feature. Harmonics from 1x to 5x rotation frequency were "
                    "scanned and dynamic eccentricity tracked healthy motors closely at "
                    "every one, so no formula is claimed. Any detection rests on the wide "
                    "0-200 Hz envelope band alone.",
        "reliability": {
            "precision": 0.25, "recall": 0.11, "f1": 0.15,
            "note": "Unreliable, and known to be. Nearly nine in ten real cases are missed "
                    "and most flags are false. Treat a flag here as noise unless something "
                    "else agrees with it.",
        },
        "causes": ["Bent shaft",
                   "Worn bearings letting the rotor orbit",
                   "Rotor imbalance",
                   "Thermal bow after uneven heating"],
        "if_ignored": "Cyclic magnetic pull shortens bearing life sharply, and severe cases "
                      "reach rotor-to-stator contact.",
        "action": "Do not act on this finding alone. Check bearing clearance and shaft "
                  "runout if other evidence points the same way.",
        "prevention": ["Balance the rotor after any rewind or repair",
                       "Replace bearings before clearance opens up",
                       "Allow even cooling after high-load runs"],
    },
    "winding": {
        "summary": "Insulation degradation or a turn-to-turn short in the stator winding.",
        "measured": "Negative-sequence current magnitude, from a symmetrical-component "
                    "transform of the three phase fundamental phasors, plus the wide "
                    "envelope band. Shorted turns unbalance the three-phase field, which "
                    "this quantity captures.",
        "reliability": {
            "precision": 0.63, "recall": 0.21, "f1": 0.31,
            "note": "Conservative: when it flags, it is more often right than wrong, but it "
                    "misses roughly four in five real cases. The absence of a winding flag "
                    "is NOT evidence of healthy insulation.",
        },
        "causes": ["Thermal ageing of insulation, accelerated by overload",
                   "Voltage transients from switching or inverter dv/dt",
                   "Moisture or contamination tracking across the winding",
                   "Mechanical abrasion from loose coils"],
        "if_ignored": "A turn-to-turn short becomes a phase-to-phase or earth fault, often "
                      "within weeks, and trips protection without warning.",
        "action": "Take out of service for insulation resistance and surge testing. This is "
                  "the finding least safe to defer.",
        "prevention": ["Thermal monitoring and correct overload protection",
                       "Output filters on inverter-fed motors",
                       "Keep windings clean and dry; check enclosure sealing"],
    },
    "voltage_unbalance": {
        "summary": "The three supply phases are unequal. A fault in the supply, not in the "
                   "motor.",
        "measured": "Negative-sequence current magnitude from the symmetrical-component "
                    "transform of the three phase fundamentals -- the same quantity used "
                    "for winding faults, but a far stronger and cleaner signal here.",
        "reliability": {
            "precision": 1.00, "recall": 1.00, "f1": 1.00,
            "note": "Perfect on the held-out validation: every real case found, no false "
                    "alarms. An unbalanced supply is directly and unambiguously visible as "
                    "asymmetry between the phase currents.",
        },
        "causes": ["Unevenly distributed single-phase loads on the same feed",
                   "A loose or corroded connection in one phase",
                   "Failing transformer tap or contactor pole",
                   "Blown fuse in a power-factor correction bank"],
        "if_ignored": "A small voltage unbalance produces a much larger current unbalance, "
                      "and negative-sequence current heats the rotor disproportionately. "
                      "Every motor on the feed ages faster.",
        "action": "Switchboard work, not motor work. Measure phase voltages at the supply "
                  "and inspect connections -- sending a motor workshop finds nothing.",
        "prevention": ["Balance single-phase loads across phases",
                       "Thermographic survey of connections on interval",
                       "Periodic phase-voltage logging at the distribution board"],
    },
    "bend": {
        "summary": "A bent shaft, so the rotor centre traces a circle as it turns.",
        "measured": "Nothing. A bent shaft produces no signature in the phase currents, and "
                    "none of the features used here respond to it.",
        "reliability": {
            "precision": 0.00, "recall": 0.00, "f1": 0.00,
            "note": "This system cannot detect a bent shaft. Validation scored zero on "
                    "every metric -- the fault shows in vibration measurements, which this "
                    "system does not take. The location exists only because the dataset "
                    "labels it. A healthy result here means nothing was measurable, not "
                    "that the shaft is straight.",
        },
        "causes": ["Impact during handling or transport",
                   "Thermal bow from uneven cooling after a hot run",
                   "Excessive belt tension or overhung load",
                   "Incorrect coupling force at installation"],
        "if_ignored": "Constant cyclic loading destroys bearings quickly and can crack the "
                      "shaft at a keyway or step.",
        "action": "Diagnose by other means -- a dial indicator on the shaft, or a vibration "
                  "measurement. Do not rely on this system for it.",
        "prevention": ["Support the shaft properly in handling and storage",
                       "Correct belt tension and coupling alignment",
                       "Even cool-down after sustained high load"],
    },
}
