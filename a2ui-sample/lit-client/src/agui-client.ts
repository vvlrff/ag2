/**
 * AG-UI SSE client that talks to the AG2 backend and extracts A2UI messages.
 */

const A2UI_TAG_RE = /<a2ui-json>\s*([\s\S]*?)\s*<\/a2ui-json>/g;

export interface AGUIResponse {
  text: string;
  a2uiMessages: any[];
}

export async function sendMessage(
  url: string,
  userMessage: string,
  threadId: string = "t1",
): Promise<AGUIResponse> {
  const runId = crypto.randomUUID();

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      threadId,
      runId,
      state: {},
      messages: [{ id: crypto.randomUUID(), role: "user", content: userMessage }],
      tools: [],
      context: [],
      forwardedProps: {},
    }),
  });

  if (!response.ok) {
    throw new Error(`Backend error: ${response.status} ${response.statusText}`);
  }

  const body = await response.text();

  // Parse SSE events
  let fullText = "";
  const allA2UIMessages: any[] = [];

  for (const line of body.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    try {
      const event = JSON.parse(line.slice(6));

      if (event.type === "TEXT_MESSAGE_CHUNK" && event.delta) {
        // Extract A2UI JSON blocks from delta
        const matches = [...event.delta.matchAll(A2UI_TAG_RE)];
        if (matches.length > 0) {
          for (const match of matches) {
            try {
              const parsed = JSON.parse(match[1]);
              if (Array.isArray(parsed)) {
                allA2UIMessages.push(...parsed);
              } else {
                allA2UIMessages.push(parsed);
              }
            } catch {}
          }
          // Keep text outside tags
          fullText += event.delta.replace(A2UI_TAG_RE, "").trim();
        } else {
          fullText += event.delta;
        }
      }

      if (event.type === "ACTIVITY_SNAPSHOT" && event.content?.operations) {
        const ops = event.content.operations;
        if (Array.isArray(ops)) {
          allA2UIMessages.push(...ops);
        }
      }

      if (event.type === "TEXT_MESSAGE_CONTENT" && event.delta) {
        fullText += event.delta;
      }
    } catch {}
  }

  return { text: fullText.trim(), a2uiMessages: allA2UIMessages };
}
