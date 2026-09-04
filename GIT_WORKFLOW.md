# Git workflow

How we use Git and branches on this project. Read this once before you push anything. The Git commands are the same on Mac and Windows, so most of this guide is shared, and the two are split only where they genuinely differ, which is installing Git, which terminal to open, line endings, and copying to the clipboard.

## The short version

- `main` always holds working code. Nobody pushes to it directly.
- You do each piece of work on its own short-lived branch, then open a pull request (PR) for a teammate to review before it merges into `main`.
- Keep branches small and merge them often, rather than saving weeks of work for one big merge.

## Placeholders

Some commands below contain a value in angle brackets. Replace the whole thing, brackets included, with your own value. Do not type the brackets. You will see these three:

- `<repo-url>` is the repository address you copy from the green Code button on GitHub.
- `<your-feature>` is a short, hyphenated description of the task you are working on, for example `data-pipeline` or `hybrid-retriever`. So a branch written here as `feature/<your-feature>` becomes `feature/data-pipeline` when you actually type it.
- `<the-file>` is the real name of a file that Git mentions in a message, which you copy from Git's own output.

So `git checkout -b feature/<your-feature>` is never typed literally. If you are building the data pipeline, you type `git checkout -b feature/data-pipeline`.

## One-time setup

> If you already have Git installed and connected to GitHub (for example through your IDE such as VS Code), skip steps 1 to 4 and go straight to step 5, Clone the repo. Do steps 1 to 4 only if you have never set up Git on this machine.

### 1. Install Git

**Mac.** Git usually comes with Apple's command line tools.

```bash
xcode-select --install    # installs Apple's command line tools, which include Git
```

Or, if you use Homebrew:

```bash
brew install git          # downloads and installs the latest Git
```

**Windows.** Download "Git for Windows" from https://git-scm.com/download/win and run the installer with the default options. This also installs **Git Bash**, a terminal where the commands match the Mac ones exactly. Use Git Bash for this guide. PowerShell also runs the same Git commands if you prefer it.

Check it worked (both):

```bash
git --version             # prints the installed version, which confirms Git is available
```

### 2. Tell Git who you are

Git stamps every commit with a name and email, so set them once. Use the same email as your GitHub account.

```bash
git config --global user.name "Your Name"        # the name shown on your commits
git config --global user.email "you@example.com" # the email that links commits to your GitHub account
```

`--global` means this applies to every repo on your machine, so you only do it once.

### 3. Set line endings

Windows and Mac mark the end of a line with different invisible characters. If this is not set, files can look completely changed in every diff even when nobody touched them.

**Mac:**

```bash
git config --global core.autocrlf input   # leave line endings as they are when committing
```

**Windows:**

```bash
git config --global core.autocrlf true    # convert line endings correctly on checkout and commit
```

### 4. Connect to GitHub

The simplest option is the GitHub CLI. Install `gh` from https://cli.github.com, then run:

```bash
gh auth login             # walks you through signing in to GitHub from the terminal
```

Follow the prompts, choosing GitHub.com, then HTTPS, then logging in through the browser. You do this once, and Git will use it for all pushes and pulls. If you would rather use SSH keys, follow GitHub's own SSH guide, but the CLI is easier for most people.

### 5. Clone the repo

Cloning downloads the whole repo into a new folder on your computer, already linked to GitHub. You do this once per machine.

First copy the repo address from the green Code button on GitHub, using the HTTPS tab. Then:

```bash
git clone <repo-url>                       # downloads the repo into a new folder named after it
cd bt4103-team8-sec-filing-assistant       # moves your terminal into that new folder
```

`cd` means "change directory", so the second line puts you inside the project so the later commands act on it.

## Branch naming

You do every piece of work on its own branch, which is a private copy of the code taken from `main`. Name it with a prefix so everyone can tell branches apart at a glance:

- `feature/<your-feature>` for new work, for example `feature/data-pipeline` or `feature/hybrid-retriever`
- `fix/<what-you-fixed>` for bug fixes, for example `fix/edgar-rate-limit`
- `docs/<what-you-documented>` for documentation, for example `docs/readme-setup`

Keep branches short-lived. Do one focused task on a branch, merge it, then start a fresh branch for the next task. Add your initials if two people might touch the same area, for example `feature/aq-chunking`.

## The daily loop

This is the cycle you repeat for every task. The four moves are branch, commit, push, and open a pull request.

### Start a new piece of work

Always branch off an up-to-date `main` so you are not building on old code.

```bash
git checkout main                       # switch to the main branch
git pull                                # download everyone's latest merged work into your main
git checkout -b feature/<your-feature>  # create a new branch and switch onto it in one step
```

The `-b` in the last line is what creates the branch. After this you are working on your own branch and `main` is untouched.

### Save your work as you go

As you finish small logical chunks, record them.

```bash
git add .                                        # stage every changed file, ready to be committed
git commit -m "Short description of what you did" # save a snapshot of the staged changes, with a message
```

`git add .` picks up all your changes, where the `.` means "this folder and everything under it". `git commit` saves them as one snapshot. Commit often in small pieces rather than one huge commit at the end. Write the message as an instruction, for example "Add EDGAR download function" rather than "added stuff".

### Push your branch to GitHub

Your commits live only on your laptop until you push them.

```bash
git push -u origin feature/<your-feature>   # upload your branch to GitHub and link it for next time
```

`origin` is the name Git uses for the GitHub copy of the repo. The `-u` links your local branch to it, and only the first push of a branch needs `-u`. After that, plain `git push` uploads any new commits.

### Open a pull request

Go to the repo on GitHub. It shows a prompt to open a pull request from your branch into `main`. Write a short description of what you changed, then request a review from a teammate. Once they approve, click "Squash and merge" so `main` keeps a clean history. GitHub then offers a button to delete the branch, which you should take.

Back in your terminal, move off the old branch and refresh your `main`:

```bash
git checkout main    # switch back to the main branch
git pull             # bring in your just-merged work plus anyone else's
```

Then start your next task from "Start a new piece of work" again.

## Keeping your branch up to date

If your branch has been open a while and teammates have merged things into `main`, pull those changes into your branch so it does not drift, and so any conflicts show up early rather than at the end.

```bash
git checkout main                       # switch to main
git pull                                # get the latest merged work
git checkout feature/<your-feature>     # switch back to your branch
git merge main                          # bring main's new changes into your branch
```

If Git reports a conflict, it means the same lines were changed in two places and Git cannot decide which to keep. Open the file it names. You will see the two versions marked like this:

```
<<<<<<< HEAD
your version
=======
their version
>>>>>>> main
```

Edit the file so it reads the way it should, then delete the three marker lines (`<<<<<<<`, `=======`, and `>>>>>>>`). Then:

```bash
git add <the-file>   # mark the conflict in that file as resolved
git commit           # complete the merge
```

If it gets messy and you want to start over, `git merge --abort` puts you back to before the merge.

## Rules that keep the repo sane

- Never commit straight to `main`. Always go through a branch and a PR.
- Never commit large data files. Keep the SEC corpus out of the repo (the `.gitignore` handles this) and commit only a small sample plus the code that fetches the rest. If we ever must version large files, we will set up Git LFS.
- Never commit secrets. API keys and tokens go in a local `.env` file that is ignored, not in the code.
- Pull `main` before you start each session so you are working on the latest code.

## Using the `.gitignore` file

A `.gitignore` file tells Git which files and folders to leave alone, so they are never staged, committed, or pushed. We use it to keep data, secrets, and local environment files out of the shared repo.

How it works:

- The file sits in the repo root, named `.gitignore`, and is itself committed, so the whole team shares the same rules.
- Each line is a pattern. `data/raw/` ignores that folder, `*.zip` ignores every file ending in `.zip`, and `.env` ignores that one exact file.
- A line starting with `!` is an exception that re-includes something. For example `!.env.example` keeps the committed template even though the `.env.*` files above it are ignored.
- Ignoring only works on files Git is not already tracking. If a file was committed before you added it to `.gitignore`, Git keeps tracking it. To stop tracking it, run `git rm --cached <the-file>` once, then commit. That removes it from the repo but leaves it on your disk.

To use ours, a ready-made `.gitignore` for this project is provided alongside this guide. Put it in the repo root, replacing the blank one that came with the repo, then commit it like any other file:

```bash
git add .gitignore                     # stage the ignore rules
git commit -m "Add project gitignore"  # save them so the whole team shares them
git push                               # upload to GitHub
```

After that, the files it lists simply will not appear when you run `git status`, which is what you want.

## Quick command reference

| What you want | Command |
|---|---|
| See what has changed | `git status` |
| See the actual changes | `git diff` |
| Update your local `main` | `git checkout main` then `git pull` |
| Start a branch | `git checkout -b feature/<your-feature>` |
| Stage everything | `git add .` |
| Commit | `git commit -m "message"` |
| Push a new branch | `git push -u origin feature/<your-feature>` |
| Push after that | `git push` |
| Switch branches | `git checkout <branch-name>` |
| List branches | `git branch` |
| Bring `main` into your branch | `git merge main` (while on your branch) |
