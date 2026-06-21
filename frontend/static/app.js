const chatFeed = document.getElementById("chat-feed");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const memoryList = document.getElementById("memory-list");
const rememberLatestButton = document.getElementById("remember-latest");
const refreshMemoriesButton = document.getElementById("refresh-memories");
const manualMemoryForm = document.getElementById("manual-memory-form");
const manualMemoryInput = document.getElementById("manual-memory-input");
const manualMemoryKind = document.getElementById("manual-memory-kind");
const messageTemplate = document.getElementById("message-template");

function appendMessage(role, content) {
  const fragment = messageTemplate.content.cloneNode(true);
  const article = fragment.querySelector(".message");
  const bubble = fragment.querySelector(".bubble");
  article.classList.add(role);
  bubble.textContent = content;
  chatFeed.appendChild(fragment);
  chatFeed.scrollTop = chatFeed.scrollHeight;
}

function renderMemories(memories) {
  memoryList.innerHTML = "";

  if (!memories.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No memories yet. Start chatting or save one explicitly.";
    memoryList.appendChild(empty);
    return;
  }

  memories.forEach((memory) => {
    const card = document.createElement("article");
    card.className = "memory-card";
    card.innerHTML = `
      <div class="memory-top">
        <span class="pill ${memory.kind}">${memory.kind}</span>
        <button class="delete-btn" data-memory-id="${memory.id}">Delete</button>
      </div>
      <p class="memory-text">${memory.content}</p>
      <div class="memory-meta">
        <span>importance ${memory.importance.toFixed(1)}</span>
        <span>confidence ${memory.confidence.toFixed(2)}</span>
      </div>
    `;
    memoryList.appendChild(card);
  });

  document.querySelectorAll(".delete-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      await fetch(`/api/memories/${button.dataset.memoryId}`, { method: "DELETE" });
      await loadMemories();
    });
  });
}

async function loadMemories() {
  const response = await fetch("/api/memories");
  const memories = await response.json();
  renderMemories(memories);
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) return;

  appendMessage("user", message);
  messageInput.value = "";
  messageInput.style.height = "34px";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const payload = await response.json();
    if (!response.ok) {
      appendMessage("assistant", payload.detail || "Something went wrong.");
      return;
    }

    appendMessage("assistant", payload.reply || "No reply returned.");
    await loadMemories();
  } catch (error) {
    appendMessage("assistant", "The app could not reach the backend. Check that the server is running.");
  }
});

rememberLatestButton.addEventListener("click", async () => {
  const response = await fetch("/api/memories/remember-latest", { method: "POST" });
  if (response.ok) {
    await loadMemories();
  }
});

refreshMemoriesButton.addEventListener("click", loadMemories);

manualMemoryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = manualMemoryInput.value.trim();
  if (!content) return;

  await fetch("/api/memories", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      content,
      kind: manualMemoryKind.value,
    }),
  });

  manualMemoryInput.value = "";
  await loadMemories();
});

messageInput.addEventListener("input", () => {
  messageInput.style.height = "34px";
  messageInput.style.height = `${messageInput.scrollHeight}px`;
});

loadMemories();
