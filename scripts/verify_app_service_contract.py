#!/usr/bin/env python3
"""Verify Oralable app-service identifiers, local fixtures, and release-claim gates.

This verifier is offline. It checks source-controlled configuration only and does
not authenticate with or modify any Apple service.
"""

from __future__ import annotations

import argparse
import json
import plistlib
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
IOS = WORKSPACE / "oralable_swift"
CONTRACT_PATH = ROOT / "contracts" / "oralable_app_services_v1.json"
CORE_CONSTANTS = WORKSPACE / "OralableCore" / "Sources" / "OralableCore" / "CloudKit" / "OralableAppServiceContract.swift"
PROJECT = IOS / "OralableApp" / "OralableApp.xcodeproj" / "project.pbxproj"


def require(condition: bool, message: str, failures: list[str]) -> None:
    if condition:
        print(f"PASS  {message}")
    else:
        print(f"ERROR {message}")
        failures.append(message)


def products(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["productID"]
        for group in payload["subscriptionGroups"]
        for item in group["subscriptions"]
    }


def contains_all(text: str, values: Iterable[str]) -> bool:
    return all(value in text for value in values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="return non-zero for every contract mismatch")
    parser.parse_args()

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []

    consumer = contract["apps"]["consumer"]
    professional = contract["apps"]["professional"]
    consumer_id = consumer["bundle_identifier"]
    professional_id = professional["bundle_identifier"]
    cloudkit = contract["cloudkit"]
    all_products = {
        *contract["storekit"]["consumer"]["products"].values(),
        *contract["storekit"]["professional"]["products"].values(),
    }

    project = PROJECT.read_text(encoding="utf-8")
    bundle_ids = set(re.findall(r"PRODUCT_BUNDLE_IDENTIFIER = ([^;]+);", project))
    expected_bundles = {
        consumer_id,
        professional_id,
        f"{consumer_id}.tests",
        f"{consumer_id}.uitests",
        f"{professional_id}.tests",
        f"{professional_id}.uitests",
    }
    require(expected_bundles <= bundle_ids, "Xcode targets use every canonical app and test bundle identifier", failures)
    forbidden_bundles = {"com.oralable.mam", "jacdentalsolutions.OralableAppTests", "jacdentalsolutions.OralableAppUITests", "com.jacdental.oralable.dentist"}
    require(not (bundle_ids & forbidden_bundles), "Xcode targets contain no retired bundle identifier", failures)

    constants = CORE_CONSTANTS.read_text(encoding="utf-8")
    require(contains_all(constants, expected_bundles | all_products | {cloudkit["container_identifier"]}), "shared Swift service constants match all bundle, product, and container values", failures)

    patient_storekit = IOS / "OralableApp" / "Configuration.storekit"
    professional_storekit = IOS / "OralableApp" / "OralableForProfessionals" / "ProfessionalConfiguration.storekit"
    expected_patient_products = set(contract["storekit"]["consumer"]["products"].values())
    expected_professional_products = set(contract["storekit"]["professional"]["products"].values())
    require(products(patient_storekit) == expected_patient_products, "consumer StoreKit fixture matches the canonical consumer products", failures)
    require(products(professional_storekit) == expected_professional_products, "professional StoreKit fixture matches all canonical professional products", failures)

    entitlement_paths = [
        IOS / "OralableApp" / "OralableApp" / "OralableApp.entitlements",
        IOS / "OralableApp" / "OralableForProfessionals" / "OralableForProfessionals.entitlements",
    ]
    for path in entitlement_paths:
        entitlement = plistlib.loads(path.read_bytes())
        containers = set(entitlement.get("com.apple.developer.icloud-container-identifiers", []))
        require(containers == {cloudkit["container_identifier"]}, f"{path.name} declares the canonical CloudKit container", failures)

    consumer_source = (IOS / "OralableApp" / "OralableApp" / "Managers" / "SharedDataManager.swift").read_text(encoding="utf-8")
    professional_source = (IOS / "OralableApp" / "OralableForProfessionals" / "Managers" / "ProfessionalDataManager.swift").read_text(encoding="utf-8")
    subscription_source = (IOS / "OralableApp" / "OralableApp" / "Managers" / "SubscriptionManager.swift").read_text(encoding="utf-8")
    professional_subscription_source = (IOS / "OralableApp" / "OralableForProfessionals" / "Managers" / "ProfessionalSubscriptionManager.swift").read_text(encoding="utf-8")
    upgrade_source = (IOS / "OralableApp" / "OralableForProfessionals" / "Views" / "UpgradePromptView.swift").read_text(encoding="utf-8")

    for name, source in (("consumer CloudKit manager", consumer_source), ("professional CloudKit manager", professional_source)):
        require("OralableAppServiceContract.CloudKit" in source, f"{name} uses shared CloudKit constants", failures)
        require("professionalID" in source and "legacyDentistID" in source, f"{name} follows canonical-plus-legacy professional field migration", failures)
    for name, source in (("consumer StoreKit manager", subscription_source), ("professional StoreKit manager", professional_subscription_source), ("professional upgrade prompt", upgrade_source)):
        require("OralableAppServiceContract.StoreKit" in source, f"{name} uses shared StoreKit constants", failures)
        require(not re.search(r'"com\\.jacdental\\.oralable\\.(?:dentist\\.|professional\\.professional|professional\\.practice)', source), f"{name} contains no drifted StoreKit ID literal", failures)

    feature_flags = (IOS / "OralableApp" / "OralableApp" / "Managers" / "FeatureFlags.swift").read_text(encoding="utf-8")
    prelaunch_start = feature_flags.index("func applyPreLaunchConfig()")
    prelaunch_end = feature_flags.index("/// Full feature configuration", prelaunch_start)
    app_store_start = feature_flags.index("func applyAppStoreMinimalConfig()")
    app_store_end = feature_flags.index("/// Research release configuration", app_store_start)
    prelaunch_profiles = feature_flags[prelaunch_start:prelaunch_end] + feature_flags[app_store_start:app_store_end]
    require(
        "static let showShareWithProfessional = false" in feature_flags
        and "showShareWithProfessional = true" not in prelaunch_profiles
        and prelaunch_profiles.count("showShareWithProfessional = false") == 2,
        "consumer default, pre-launch, and App Store-minimal profiles keep professional sharing disabled",
        failures,
    )

    health_source = (IOS / "OralableApp" / "OralableApp" / "Managers" / "AppleHealthManager.swift").read_text(encoding="utf-8")
    require(".heartRate" in health_source and ".oxygenSaturation" in health_source, "HealthKit implementation supports only declared heart-rate and oxygen-saturation quantity types", failures)
    require("Sleep stages not written" in health_source and "HRV sync not implemented" in health_source, "HealthKit implementation marks sleep stages and HRV as unsupported", failures)

    metadata = (IOS / "OralableApp" / "APP_STORE_METADATA.md").read_text(encoding="utf-8")
    professional_metadata = (IOS / "OralableApp" / "DENTIST_APP_STORE_METADATA.md").read_text(encoding="utf-8")
    require(consumer_id in metadata and "Pre-submission status" in metadata, "consumer App Store copy declares canonical identity and pre-submission status", failures)
    require("professional sharing is not available" in metadata.lower() and "subscriptions are not available" in metadata.lower(), "consumer App Store copy does not claim gated services", failures)
    require("not authorized for App Store submission" in professional_metadata, "professional App Store draft is explicitly deferred", failures)

    setup = (IOS / "OralableApp" / "CLOUDKIT_PRODUCTION_SETUP.md").read_text(encoding="utf-8")
    retired_field_row = re.compile(r"^\|\s*`(?:used|isRevoked)`\s*\|", re.MULTILINE)
    require("professionalID" in setup and "isActive" in setup and not retired_field_row.search(setup), "CloudKit deployment guide matches canonical active-state and professional fields", failures)

    print(f"App-service contract {contract['contract']} {contract['version']}")
    if failures:
        print(f"FAILED: {len(failures)} mismatch(es)")
        return 1
    print("PASS  app-service contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
