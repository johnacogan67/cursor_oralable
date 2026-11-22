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

## Action plan

This is the path to the lock above. It is the minimum product. Steps 2 and 3 can run before the reference nights. Steps 8 to 10 wait on steps 4 to 7. Step 1 sits under all of them.

Already in hand: the five-name night bar, a Core ML score for each second (quiet, tonic, phasic, rescue), and the rule that phasic stays motion until a higher bar is cleared. The model was fit on short structured locks. A night still asks for Worn on temple, a fit check, a calibration of about 10 minutes, and a movement threshold.

1. Keep a night on the temple without a new setting. A worn night has to stay recording long enough to score. Putting the pad on stays the person’s only physical job.
2. Let the clip decide it is on the temple. Use the optical signal to mark on-temple and off-temple. When that holds on worn and unworn checks, remove the Worn on temple switch.
3. Take the baseline automatically. Replace the 10-minute calibration with a short baseline at the start of the night, once the signal is good. Remove the fit wizard when that baseline is stable without a person running it.
4. Collect the clench and grind reference. One overnight with the temple clip and ANR electrical sensors on the same muscle. Score tonic against sustained muscle bursts, and phasic against those bursts or against tooth-sound audio. Report the hit rate and the lag. The six-minute lab pack does not count.
5. Collect the apnea-like reference. One overnight with the temple clip and AcuPebble, or a sleep study, on one clock. Score how often a high rescue probability lands on the scored event ends. Write the miss. Do not compute an apnea count from the temple recording.
6. Collect a sleep–wake reference. On a scored night, compare hours asleep with hours worn. Until that check exists, do not call the bar a sleep record.
7. Agree the three bars before any new wording appears. Fix the held-out level for tonic, a higher level for phasic, and the rescue agreement. Below the phasic bar the lane stays motion. A weak phasic match is an allowed result.
8. Train the same Core ML model on those overnight labels. Keep the four classes. Hold out people the model never saw. The phone reports a probability. It prints clench, grind, or the rescue agreement only when the probability clears the bar from step 7.
9. Remove the movement threshold. Do this only after phasic comes from that model and the reference nights, not from a number in settings.
10. Count the three confidences only inside scored sleep. Quiet and Recovery stay as they are. The screen is still the five-name bar.

## What this does not say

No kit has shipped. This note does not close a shortcoming in [DENTIST_APPROXIMATE.md](./DENTIST_APPROXIMATE.md). It does not equate tonic with scored muscle events, phasic with grinding, rescue with an apnea, or temple oxygen with an apnea index.
