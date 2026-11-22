# Recognition confidence — clench, grind, apnea-like pattern

**As at:** 22 Nov 2026  
**Status:** Goal lock. Not a result. Not a clearance.  
**Sits beside:** [MEASUREMENT_CONSTRUCT_MAP.md](./MEASUREMENT_CONSTRUCT_MAP.md)

**One-liner:** One Core ML model, three references, three thresholds. The night bar keeps its five names. The person sets as little as possible, and the end state is that they set nothing. Confidence does not rename that bar, and it does not produce an apnea count.

---

## The screen stays

The night record is **Quiet**, **Tonic**, **Phasic**, **Rescue**, and **Recovery**. That is the screen this goal keeps.

| Lane | What it is now | What a high confidence may add later |
|------|----------------|--------------------------------------|
| Quiet | Calm stretch | Stays quiet |
| Tonic | Jaw-load lane | Clench, after an overnight electrical match |
| Phasic | Motion lane | Grind only above a higher bar. Below that bar it stays motion |
| Rescue | Jaw load with a low temple oxygen reading | How often those bars land near AcuPebble or sleep-study events |
| Recovery | Short settle after an event | Stays recovery |

## User configuration

Configuration is a step on the way, not part of the finished night. Add no new setting. Retire the ones that exist.

Today a night still asks for three choices: Worn on temple, a temple fit check, and a calibration of about 10 minutes. A movement threshold also sits in settings. Putting the pad on the temple is wear, not a setting. That stays.

1. Keep those asks only while the clip cannot tell a good temple signal from a bench signal by itself.
2. Drop Worn on temple once the optical signal marks on-temple and off-temple without a switch.
3. Turn the 10-minute calibration into a short baseline taken automatically at the start of the night.
4. Remove the movement threshold from settings once phasic uses the model and the reference nights, not a number the person picks.
5. Stop when a night starts because the pad is on and the signal is good. No switch, no fit wizard, and no threshold screen.

Less configuration does not replace the reference matches below.

## The lock

Clench is reported when the tonic probability is high and a held-out overnight match to temporalis electrical sensors clears the agreed level.

Grind is reported only above a higher bar, against those sensors or tooth-sound audio. Below that bar the lane stays motion.

An apnea-like pattern is reported as how often a high rescue probability lands on AcuPebble or sleep-study events. The temple recording does not produce an apnea count.

Hours worn are not hours asleep until a scored sleep–wake check exists.

The three references are the ANR electrical sensors (or a later sleep study with audio and video) for tonic and phasic, tooth-sound audio as the other phasic reference, and AcuPebble or a sleep study for rescue. The model classes today are quiet, tonic, phasic, and rescue. They were fit on short structured locks. They are not yet a reported confidence against an overnight reference.

## What this does not say

No kit has shipped. This note does not close a shortcoming in [DENTIST_APPROXIMATE.md](./DENTIST_APPROXIMATE.md). It does not equate tonic with scored muscle events, phasic with grinding, rescue with an apnea, or temple oxygen with an apnea index.
