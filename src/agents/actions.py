"""
Action Map — Predefined, parameterized test step actions.

Each action has:
- A human-readable template with {placeholders}
- Parameter definitions (what each placeholder means)
- A Playwright code template for execution

The designer agent uses these actions when generating test cases,
producing structured steps that are both readable and executable.
"""

ACTIONS = {
    # --- Interaction ---
    "click": {
        "template": "Click on {element}",
        "description": "Click a button, link, or any clickable element",
        "params": {
            "element": "Element description (e.g., 'Submit button', 'Login link')",
        },
        "playwright": "await page.get_by_role('{role}', name='{name}').click()",
    },
    "fill": {
        "template": "Fill {element} with {value}",
        "description": "Type text into an input field or textarea",
        "params": {
            "element": "Input field description (e.g., 'Email textbox', 'Search input')",
            "value": "Text to type",
        },
        "playwright": "await page.get_by_role('{role}', name='{name}').fill('{value}')",
    },
    "clear": {
        "template": "Clear {element}",
        "description": "Clear the contents of an input field",
        "params": {
            "element": "Input field description",
        },
        "playwright": "await page.get_by_role('{role}', name='{name}').clear()",
    },
    "select": {
        "template": "Select {value} from {element}",
        "description": "Choose an option from a dropdown/select",
        "params": {
            "element": "Dropdown description (e.g., 'Country dropdown')",
            "value": "Option to select",
        },
        "playwright": "await page.get_by_role('{role}', name='{name}').select_option('{value}')",
    },
    "check": {
        "template": "Check {element}",
        "description": "Check a checkbox or toggle",
        "params": {
            "element": "Checkbox description (e.g., 'Remember me checkbox')",
        },
        "playwright": "await page.get_by_role('{role}', name='{name}').check()",
    },
    "uncheck": {
        "template": "Uncheck {element}",
        "description": "Uncheck a checkbox or toggle",
        "params": {
            "element": "Checkbox description",
        },
        "playwright": "await page.get_by_role('{role}', name='{name}').uncheck()",
    },
    "hover": {
        "template": "Hover over {element}",
        "description": "Move the mouse over an element to trigger hover state",
        "params": {
            "element": "Element description",
        },
        "playwright": "await page.get_by_role('{role}', name='{name}').hover()",
    },
    "press_key": {
        "template": "Press {key} on {element}",
        "description": "Press a keyboard key (Enter, Tab, Escape, etc.)",
        "params": {
            "element": "Element that has focus",
            "key": "Key name (Enter, Tab, Escape, ArrowDown, etc.)",
        },
        "playwright": "await page.get_by_role('{role}', name='{name}').press('{key}')",
    },
    "upload_file": {
        "template": "Upload {file} to {element}",
        "description": "Upload a file via a file input",
        "params": {
            "element": "File input description",
            "file": "File path or name to upload",
        },
        "playwright": "await page.get_by_role('{role}', name='{name}').set_input_files('{file}')",
    },

    # --- Navigation ---
    "navigate": {
        "template": "Navigate to {url}",
        "description": "Go to a specific URL",
        "params": {
            "url": "Target URL",
        },
        "playwright": "await page.goto('{url}')",
    },
    "go_back": {
        "template": "Go back to previous page",
        "description": "Navigate back in browser history",
        "params": {},
        "playwright": "await page.go_back()",
    },
    "reload": {
        "template": "Reload the page",
        "description": "Refresh the current page",
        "params": {},
        "playwright": "await page.reload()",
    },

    # --- Waiting ---
    "wait_for_element": {
        "template": "Wait for {element} to be {state}",
        "description": "Wait until an element reaches a specific state",
        "params": {
            "element": "Element description",
            "state": "visible | hidden | enabled | disabled",
        },
        "playwright": "await page.get_by_role('{role}', name='{name}').wait_for(state='{state}')",
    },
    "wait_for_url": {
        "template": "Wait for URL to contain {url_part}",
        "description": "Wait for navigation to a URL pattern",
        "params": {
            "url_part": "Expected URL substring or pattern",
        },
        "playwright": "await page.wait_for_url('**/{url_part}**')",
    },

    # --- Assertions ---
    "assert_visible": {
        "template": "Verify {element} is visible",
        "description": "Assert that an element is visible on the page",
        "params": {
            "element": "Element description",
        },
        "playwright": "await expect(page.get_by_role('{role}', name='{name}')).to_be_visible()",
    },
    "assert_hidden": {
        "template": "Verify {element} is not visible",
        "description": "Assert that an element is hidden or removed",
        "params": {
            "element": "Element description",
        },
        "playwright": "await expect(page.get_by_role('{role}', name='{name}')).to_be_hidden()",
    },
    "assert_text": {
        "template": "Verify {element} contains text {expected}",
        "description": "Assert that an element contains specific text",
        "params": {
            "element": "Element description",
            "expected": "Expected text content",
        },
        "playwright": "await expect(page.get_by_role('{role}', name='{name}')).to_contain_text('{expected}')",
    },
    "assert_value": {
        "template": "Verify {element} has value {expected}",
        "description": "Assert that an input has a specific value",
        "params": {
            "element": "Element description",
            "expected": "Expected input value",
        },
        "playwright": "await expect(page.get_by_role('{role}', name='{name}')).to_have_value('{expected}')",
    },
    "assert_enabled": {
        "template": "Verify {element} is enabled",
        "description": "Assert that an element is not disabled",
        "params": {
            "element": "Element description",
        },
        "playwright": "await expect(page.get_by_role('{role}', name='{name}')).to_be_enabled()",
    },
    "assert_disabled": {
        "template": "Verify {element} is disabled",
        "description": "Assert that an element is disabled",
        "params": {
            "element": "Element description",
        },
        "playwright": "await expect(page.get_by_role('{role}', name='{name}')).to_be_disabled()",
    },
    "assert_url": {
        "template": "Verify URL contains {expected}",
        "description": "Assert the current page URL",
        "params": {
            "expected": "Expected URL or substring",
        },
        "playwright": "await expect(page).to_have_url(re.compile(r'.*{expected}.*'))",
    },
    "assert_count": {
        "template": "Verify {element} appears {count} times",
        "description": "Assert the number of matching elements",
        "params": {
            "element": "Element description",
            "count": "Expected count",
        },
        "playwright": "await expect(page.get_by_role('{role}', name='{name}')).to_have_count({count})",
    },

    # --- Scroll ---
    "scroll_to": {
        "template": "Scroll to {element}",
        "description": "Scroll the page until an element is in view",
        "params": {
            "element": "Element description",
        },
        "playwright": "await page.get_by_role('{role}', name='{name}').scroll_into_view_if_needed()",
    },
    "scroll_page": {
        "template": "Scroll page {direction}",
        "description": "Scroll the page up or down",
        "params": {
            "direction": "down | up",
        },
        "playwright": "await page.evaluate('window.scrollBy(0, {delta})')",
    },
}


def get_action_catalog_prompt() -> str:
    """Build a prompt fragment listing all available actions for the LLM."""
    lines = ["Available actions for test steps:\n"]
    for action_id, action in ACTIONS.items():
        params = ", ".join(
            f"{{{k}}}: {v}" for k, v in action["params"].items()
        )
        lines.append(f"- **{action_id}**: {action['template']}")
        if params:
            lines.append(f"  Params: {params}")
    return "\n".join(lines)


def get_action_ids() -> list[str]:
    """Return all valid action IDs."""
    return list(ACTIONS.keys())


def render_step(action_id: str, params: dict) -> str:
    """Render a human-readable step from an action + params."""
    action = ACTIONS.get(action_id)
    if not action:
        return f"[unknown action: {action_id}] {params}"
    template = action["template"]
    for key, value in params.items():
        template = template.replace(f"{{{key}}}", str(value))
    return template
