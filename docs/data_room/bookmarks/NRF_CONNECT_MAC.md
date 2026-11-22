# nRF Connect on the Mac

**As at:** 20 Nov 2026  
**Tone:** where to get the Mac app. Not a firmware result and not a clearance claim.  
**Related:** [OTA_DEVICE_MANAGER.md](../../../../oralable_nrf/docs/OTA_DEVICE_MANAGER.md)

**One-liner:** The Oralable iPhone project runs from Xcode. nRF Connect does not. Nordic keeps that source private and posts a Mac disk image in the iOS repo.

---

## Bookmarks

| # | Source | Why it matters |
|---|--------|----------------|
| 1 | [Releases folder](https://github.com/nordicsemi/IOS-nRF-Connect/tree/master/Releases) | Mac builds live here. The App Store copy is for iPhone and iPad. |
| 2 | [nRF Connect 2.8.1 disk image](https://github.com/nordicsemi/IOS-nRF-Connect/blob/master/Releases/nRF-Connect-2-8-1-b1127.dmg) | The file their readme links on 20 Nov 2026. Build 1127. Needs macOS 14 or later. Look in the folder first in case a newer image is there. |

The iOS Simulator has no radio, so it cannot see the MAM. Quit nRF Connect on the phone before the Mac app connects, or the module stays tied to the phone.

## Claim discipline

- Do not say this Mac app is the App Store iPhone binary.
- Do not treat a Mac scan as a Gen 2 gate pass.
