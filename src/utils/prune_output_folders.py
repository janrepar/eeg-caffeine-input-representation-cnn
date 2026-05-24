import argparse
import re
import shutil
from datetime import datetime, date
from pathlib import Path


DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")


def extract_date_from_folder_name(folder_name: str) -> date | None:
    """
    Extracts first date in format YYYY-MM-DD from folder name.

    Examples:
        raw_eegnetlike_2026-05-22_204449 -> 2026-05-22
        model_comparison_loso_5folds_1valsubj_2026-05-23_115000 -> 2026-05-23
    """

    match = DATE_PATTERN.search(folder_name)

    if match is None:
        return None

    date_string = match.group(0)

    return datetime.strptime(date_string, "%Y-%m-%d").date()


def find_old_folders(parent_folder: Path, before_date: date):
    """
    Finds direct child folders whose date in folder name is older than before_date.
    """

    folders_to_delete = []

    for child in parent_folder.iterdir():
        if not child.is_dir():
            continue

        folder_date = extract_date_from_folder_name(child.name)

        if folder_date is None:
            print(f"Skipping folder without date: {child}")
            continue

        if folder_date < before_date:
            folders_to_delete.append((child, folder_date))

    return folders_to_delete


def main():
    parser = argparse.ArgumentParser(
        description="Delete output folders older than a given date."
    )

    parser.add_argument(
        "--folder",
        required=True,
        help="Parent folder in which old subfolders should be deleted, e.g. outputs/results",
    )

    parser.add_argument(
        "--before",
        required=True,
        help="Delete folders with date lower than this date. Format: YYYY-MM-DD",
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete folders. Without this flag, script only prints what would be deleted.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show folders that would be deleted.",
    )

    args = parser.parse_args()

    parent_folder = Path(args.folder)

    if not parent_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {parent_folder}")

    if not parent_folder.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {parent_folder}")

    before_date = datetime.strptime(args.before, "%Y-%m-%d").date()

    folders_to_delete = find_old_folders(
        parent_folder=parent_folder,
        before_date=before_date,
    )

    print("\n" + "=" * 100)
    print(f"Parent folder: {parent_folder}")
    print(f"Delete folders with date lower than: {before_date}")
    print("=" * 100)

    if not folders_to_delete:
        print("No folders found for deletion.")
        return

    print("\nFolders matched:")
    for folder, folder_date in folders_to_delete:
        print(f"  {folder_date}  ->  {folder}")

    print("\nTotal folders matched:", len(folders_to_delete))

    if args.dry_run or not args.delete:
        print("\nDRY RUN ONLY. Nothing was deleted.")
        print("To actually delete, run again with --delete")
        return

    confirm = input("\nType DELETE to confirm deletion: ")

    if confirm != "DELETE":
        print("Deletion cancelled.")
        return

    for folder, folder_date in folders_to_delete:
        print(f"Deleting: {folder}")
        shutil.rmtree(folder)

    print("\nDone. Deleted folders:", len(folders_to_delete))


if __name__ == "__main__":
    main()