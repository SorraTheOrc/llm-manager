<!-- WORKFLOW: start -->
## Workflow for AI Agents

It is expected that a session will be started by a human operator who will supply an initial prompt which defines the overall goals and context for the work to be done. When receiving such a prompt the agent will create an initial work-item in the worklog to track the work required to meet those goals. The work-item is created with a command such as `wl create "<work-item-title>" --description "<detailed-description-of-goals-and-context>" --issue-type <type-of-work-item> --json` (see [Work-Item Management](#work-item-management) below for more information). Remember the work-item id that is returnedm this will be referred to below as the <base-item-id>.

Once the item has been created the agent should display the outputs of `wl show <base-item-id> --format full` and confirm with the operator that the work-item accurately reflects the goals and context provided. If the operator requests changes to the work-item the agent should update the work-item description and acceptance criteria accordingly `wl update <id> --description "<updated-description>"`. DO NOT remove existing content unless it is incorrect, ONLY add to it with appropriate clarifications.

Once approved the agent should ask if they may ask further clarifying questions if required during the planning and implementation of <base-item-id>, the agent should make it clear that if the operator says no the agent will attempt to complete the task without further guidance, but being able to ask questions increases the chances of success. The agent will wait for confirmation from the operator before proceeding and remember the response.

The agent(s) will then plan and execute the work required to meet those goals by following the steps below.

0. **Claim the work-item** created by the operator:
   - Claim it with `wl update <id> --status in-progress --assignee @your-agent-name`
1. **Ensure the work-item is clearly defined**:
   - Review the description, acceptance criteria, and any related files/paths in the work item description and comments (retrieved with `wl show <id> --children --json`)
   - Review any existing work-items in the repository that may be related to this work-item (`wl search <search-terms> --json` and `wl show <id> --children --json`).
   - If the work-item is not clearly defined (it _MUST_ included a clear description of the goal and how it will change behaviour, preferably in the form of a user story, along with acceptance criteria that can be used to verify completion and references to important specifications, user-stories, designs, or other important context):
     - Search the worklog (`wl search <search-terms> --json` and `wl show <id> --children --json`) and repository for any existing information that may clarify the requirements
     - If the operator has allowed further questions ask for clarification on specific requirements, acceptance criteria, and context. Where possible provide suggested responses, but always allow for a free form text response.
     - If the operator has not allowed further questions attempt to clarify the requirements based on the existing information in the repository and worklog.
     - Update the work-item description and acceptance criteria with any clarifications found `wl update <id> --description "<updated-description>"`. DO NOT remove existing content unless it is incorrect, ONLY add to it with appropriate clarifications.
   - Once the work-item is clearly defined update its stage to `intake_complete` using `wl update <id> --stage intake_complete`
   - Report back to the operator summarising any clarifications made and proceed to the next step.
2. **Plan the work**:
   - Break down the work into smaller sub-tasks if necessary
   - Each sub-task should be a discrete unit of work that can be completed independently, if a sub-task is still too large break it down further with sub-tasks of its own
   - Verify and if possible improve the description of the goal and how it will change behaviour, preferably in the form of a user story
   - Verify and if possible improve the references to important specifications, user-stories, designs, or other important context
   - Verify and if possible improve the acceptance criteria so they are clear, measurable, and testable
   - Create child work-items for each sub-task using `wl create -t "<sub-task-title>" -d "<detailed-description>" --parent <base-item-id> --issue-type <type-of-work-item> --priority <critical|high|medium|low> --json`
   - Once planning is complete update the parent work-item stage to `plan_complete` using `wl update <base-item-id> --stage plan_complete`
   - Report back to the operator summarising the plan using `wl show <base-item-id> --children` and proceed to the next step.
3. **Decide what to work on next**:
   - Use `wl next --json` to get a recommendation for the next work-item to work on. The id of this item will be referred to below as <WIP-id>.
   - If the recommended work-item has no children proceed to the next step.
   - If the recommended work-item has children claim this work-item and mark it as in progress using `wl update <WIP-id> --status in-progress --assignee @your-agent-name`
   - Repeat this step to get the next recommended work-item until a leaf work-item (one with no children) is reached.
   - if there are no descendents of <base-item-id> left to work on go to the `End session` step.
   - Report back to the operator summarising the selected work-item and proceed to the next step.
4. **Implement the work-item**:
   - Review the content of the selected work-item
   - Review the description, acceptance criteria, and any related files/paths in the work item description and comments (retrieved with `wl show <WIP-id> --children --json`)
   - Review any existing work-items in the repository that may be related to this work-item (`wl search <search-terms> --json` and `wl show <id> --children --json`).
   - If the work-item is not clearly defined:
     - Search the worklog (`wl search <search-terms> --json` and `wl show <id> --children --json`) and repository for any existing information that may clarify the requirements
     - If the operator has allowed further questions ask for clarification on specific requirements, acceptance criteria, and context. Where possible provide suggested responses, but always allow for a free form text response.
     - If the operator has not allowed further questions attempt to clarify the requirements based on the existing information in the repository and worklog.
     - Update the work-item description and acceptance criteria with any clarifications found with `wl update <WIP-id> --description "<updated-description>"`. DO NOT remove existing content unless it is incorrect, ONLY add to it with appropriate clarifications.
   - Create a new branch for the work-item following the branch naming conventions (e.g. `wl-<WIP-id>-short-description`)
   - Complete all work required to meet the acceptance criteria (code, tests, documentation, etc.)
     - If new work-items are discovered during implementation create new work-items using `wl create "<work-item-title>" --description "<detailed-description-of-goals-and-context>" --issue-type <type-of-work-item> --json`. If the item must be completed in order to satisfy the requirements of the parent work-item, make the new item a child of the parent work-item using `--parent <parent-id>`. If it is an optional item make it a sibling of the <base-item-id> and add a reference to the base item in the description using `discovered-from:<base-item-id>`.
     - Regularly build and run all tests and checks to ensure nothing is broken
       - If the build or any tests/checks fail, fix the issues and repeat until all tests/checks pass
     - Commit changes whenever the Producer observes that a significant amount of progress has been made (ask if you think it is due), use clear commit messages that reference the WIP id and summarise the changes made.
   - If a particularly complex issue is identified or a significant design decisions or assumption is made record this in a comment on the work-item using `wl comment add <WIP-id> --comment "<detailed-comment>" --author @your-agent-name --json`
   - Once the acceptance criteria of <WIP-id> has been satisfied and all tests pass, Commit final changes to the branch with a message such as `<WIP-id>: Completed work to satisfy acceptance criteria: <acceptance-criteria-summary>`
   - When work is complete record a comment on the work-item summarising the changes made and the reason for them, including the commit hash using `wl comment add <id> --comment "Completed work, see commit <commit-hash> for details." --author @your-agent-name --json`
   - Update the work-item stage to `in_review` using `wl update <WIP-id> --stage in_review`
   - Report back to the operator summarising the work completed and proceed to the next step.
5. **Merge work into main**:
   - Update the branch to bring it into line with main
     - resolve any conflicts that arise
   - Build the application and run all tests and checks to ensure nothing is broken
     - If the build failes or any tests/checks fail, fix the issues and repeat until all tests/checks pass
   - Push the branch to the remote repository
   - Switch back to main, merge the branch and push the updated main branch to the remote repository
   - Close the work-item with a comment summarising the changes made and the reason for them, including the commit hash using `wl close <WIP-id> --reason "Completed work, see merge commit <merge-commit-hash> for details." --json`
   - Proceed to the next step.
6. **Update the operator**:
   - Provide the operator a summary of the work completed, including any relevant links (work-item id, commit hashes, PR links, etc.)
   - Do not suggest next steps at this point, simply report what has been done and proceed to the next step.
7. **Repeat**:
   - Go back to the `Decide what to work on next` step.
8. **End session**:
   - When there are no descendents of <base-item-id> left to work on, inform the operator that all required work is complete and summarize any discovered tasks, or pre-existing tasks in the worklog (`wl list --json`).
   - Ask the operator if they would like to address any of these remaining tasks now or if they would like to end the session.
   - If the operator wishes to address any remaining tasks, return to the `Claim the work-item` with the selected work-item id as the new <base-item-id>.
   - When the operator indicates that the session is complete, ensure all work-items created or worked on during the session are in the `in_review` or `done` stage.
   - Provide a final summary to the operator of all work completed during the session, including work-item ids, commit hashes, and any relevant links.
   - Thank the operator and end the session.
<!-- WORKFLOW: end -->

Follow the global AGENTS.md in addition to the rules below. The local rules below take priority in the event of a conflict.

<!-- Start base Worklog AGENTS.md file -->

## work-item Tracking with Worklog (wl)

IMPORTANT: This project uses Worklog (wl) for ALL work-item tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

## CRITICAL RULES

- Use Worklog (wl), described below, for ALL task tracking, do NOT use markdown TODOs, task lists, or other tracking methods
- When mentioning a work item always use its title followed by its ID in parentheses, e.g. "Fix login bug (WL-1234)"
- Always keep work items up to date with accurate status, priority, stage, and assignee
- Whenever you are provided with, or discover, a new work item create it in wl immediately
- Whenever you are provided with or discover important context (specifications, designs, user-stories) ensure the information is added to the description of the relevant work item(s) OR create a new work item if none exist
- Whenever you create a planning document (PRD, spec, design doc) add references to the document in the description of any work item that is directly related to the document
- Work items cannot be closed until all child items are closed, all blocking dependencies resolved and a Producer has reviewed and approved the work
- Never commit changes without associating them with a work item
- Never commit changes without ensuring all tests and quality checks pass
- Whenever a commit is made add a comment to impacted the work item(s) describing the changes, the files affected, and including the commit hash.
- If push fails, resolve and retry until it succeeds
- When using backticks in arguments to shell commands, escape them properly to avoid errors

### Important Rules

- Use wl as a primary source of truth, only the source code is more authoritative
- Always use `--json` flag for programmatic use
- When new work items are discovered or prompted while working on an existing item create a new work item with `wl create`
  - If the item must be completed before the current work item can be completed add it as a child of the current item (`wl create --parent <current-work-item-id>`)
  - If the item is related to the current work item but not blocking its completion add a reference to the current item in the description (`discovered-from:<current-work-item-id>`)
- Check `wl next` before asking "what should I work on?" and always offer the response as a next steps suggestion, with an explanation
- Run `wl --help` and `wl <cmd> --help` to learn about the capabilities of WorkLog (wl) and discover available flags
- Use work items to track all significant work, including bugs, features, tasks, epics, chores
- Use clear, concise titles and detailed descriptions for all work items
- Use parent/child relationships to track dependencies and subtasks
- Use priorities to indicate the importance of work items
- Use stages to track workflow progress
- Do NOT clutter repo root with planning documents

### work-item Types

Track work-item types with `--issue-type`:

- bug - Something broken
- feature - New functionality
- task - Work item (tests, docs, refactoring)
- epic - Large feature with subtasks
- chore - Maintenance (dependencies, tooling)

### Work Item Descriptions

- Use clear, concise titles summarizing the work item.
- Do not escape special characters
- The description must provide sufficient context for understanding and implementing the work item.
- At a minimum include:
  - A summary of the problem or feature.
  - Example User Stories if applicable.
  - Expected behaviour and outcomes.
  - Steps to reproduce (for bugs).
  - Suggested implementation approach if relevant.
  - Links to related work items or documentation.
  - Measurable and testable acceptance criteria.

### Priorities

Worklog uses named priorities:

- critical - Security, data loss, broken builds
- high - Major features, important bugs
- medium - Default, nice-to-have
- low - Polish, optimization

### Dependencies

Use parent/child relationships to track blocking dependencies.

- Child items must be completed before the parent can be closed.
- If a work item blocks another, make it a child of the blocked item.
- If a work item blocks multiple items, create the parent/child relationships with the highest priority item as the parent unless one of the items is in-progress, in which case that item should be the parent.
  - If in doubt raise for product manager review.

For non-hierarchical blocking relationships, prefer dependency edges over description-based conventions. Dependency edges are the recommended way to track blockers:

```bash
wl dep add <dependent-work-item-id> <prereq-work-item-id>
wl dep list <work-item-id> --json
wl dep rm <dependent-work-item-id> <prereq-work-item-id>
```

Description-based conventions (`discovered-from:<work-item-id>`, `related-to:<work-item-id>`, `blocked-by:<work-item-id>`) remain supported for informal cross-references and planning notes, but dependency edges should be used for any relationship that affects scheduling or blocking.

### Workflow management

- Use the `--stage` flag to track workflow stages according to your particular process,
  - e.g. `idea`, `prd_complete`, `milestones_defined`, `plan_complete`, `in_progress``done`.
- Use the `--assignee` flag to assign work items to agents.
- Use the `--tags` flag to add arbitrary tags for filtering and organization. Though avoid over-tagging.
- Use comments to document progress, decisions, and context.
- Use `risk` and `effort` fields to track complexity and potential issues.
  - If available use the `effort_and_risk` agent skill to estimate these values.

1. Check ready work: `wl next`
2. Claim your task: `wl update <id> --status in-progress`
3. Work on it: implement, test, document
4. Discover new work? Create a linked issue:

- `wl create "Found bug" --priority high --tags "discovered-from:<parent-id>"`

5. Complete: `wl close <id> --reason "PR #123 merged"`
6. Sync: run `wl sync` before ending the session

### Work-Item Management

```bash
# Create work items
wl create --help  # Show help for creating work items
wl create --title "Bug title" --description "<details>" --priority high --issue-type bug --json
wl create --title "Feature title" --description "<details>" --priority medium --issue-type feature --json
wl create --title "Epic title" --description "<details>" --priority high --issue-type epic --json
wl create --title "Subtask" --parent <parent-id> --priority medium --json
wl create --title "Found bug" --priority high --tags "discovered-from:WL-123" --json

# Update work items
wl update --help  # Show help for updating work items
wl update <work-item-id> --status in-progress --json
wl update <work-item-id> --priority high --json

# Comments
wl comment --help  # Show help for comment commands
wl comment list <work-item-id> --json
wl comment show <work-item-id>-C1 --json
wl comment update <work-item-id>-C1 --comment "Revised" --json
wl comment delete <work-item-id>-C1 --json

# Close or delete
# wl close: provide -r reason for closing; can close multiple ids
wl close <work-item-id> --reason "PR #123 merged" --json
wl close <work-item-id-1> <work-item-id-2> --json

# *Destructive command ask for confirmation before running* Dekete a work item permanently
wl delete <work-item-id> --json

# Dependencies
wl dep --help  # Show help for dependency commands
wl dep add <dependent-work-item-id> <prereq-work-item-id>
wl dep list <work-item-id> --json
wl dep rm <dependent-work-item-id> <prereq-work-item-id>
```

### Project Status

```bash
# Show the next ready work items (JSON output)
# Display a recommendation for the next item to work on in JSON
wl next --json
# Display a recommendation for the next item assigned to `agent-name` to work on
wl next --assignee "agent-name" --json
# Display a recommendation for the next item to work on that matches a keyword (in title/description/comments)
wl next --search "keyword" --json

# Show all items with status `in-progress` in JSON
wl in-progress --json
# Show in-progress items assigned to `agent-name`
wl in-progress --assignee "agent-name" --json

# Show recently created or updated work items
wl recent --json
# Show the 10 most recently created or updated items
wl recent --number 10 --json
# Include child/subtask items when showing recent items
wl recent --children --json

# List all work items except those in a completed state
wl list --json
# Limit list output
wl list -n 5 --json
# List items filtered by status (open, in-progress, closed, etc.)
wl list --status open --json
# List items filtered by priority (critical, high, medium, low)
wl list --priority high --json
# List items filtered by comma-separated tags
wl list --tags "frontend,bug" --json
# List items filtered by assignee (short or full name)
wl list --assignee alice --json
# List items filtered by stage (e.g. triage, review, done)
wl list --stage review --json

# Full-text search across all work items (title, description, comments, tags)
wl search <keywords> --json
# Search with status filter
wl search <keywords> --status open --json
# Search by work item ID (exact, unprefixed, or partial >= 8 chars)
wl search <work-item-id> --json
wl search <id-without-prefix> --json

# Show details for a specific work item
wl show <work-item-id> --comments --json
# Show details including child/subtask items
wl show <work-item-id> --children --json
```

#### Team

```bash
 # Sync local worklog data with the remote (shares changes)
 wl sync
 # Import issues from GitHub into the worklog (GitHub -> worklog)
 wl github import
 # Push worklog changes to GitHub issues (worklog -> GitHub)
 wl github push
```

#### Plugins

Depending on your setup, you may have additional wl plugins installed. Check available plugins with `wl --help` (See plugins section) to view more information about the features provided by each plugin run `wl <plugin-command> --help`

#### Help

Run `wl --help` to see general help text and available commands.
Run `wl <command> --help` to see help text and all available flags for any command.

<!-- End base Worklog AGENTS.md file -->
