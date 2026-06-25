document.addEventListener("DOMContentLoaded", () => {
  // Inject HTML into the page
  const widgetContainer = document.createElement("div");
  widgetContainer.id = "athleteedge-chat-widget";
  widgetContainer.innerHTML = `
    <div id="chat-window">
      <div class="chat-header">
        <div>
          <h3>AthleteEdge Copilot</h3>
          <p>AI Sports Doctor & Nutritionist</p>
        </div>
        <button class="close-chat-btn" id="close-chat">&times;</button>
      </div>
      <div class="chat-body" id="chat-body">
        <div class="chat-message bot">Hello! I'm your AthleteEdge AI Copilot. I can see your latest risk score and meal plans if you've run them. How can I help you today?</div>
        <div class="typing-indicator" id="typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
      <div class="chat-input-area">
        <input type="text" id="chat-input" placeholder="Ask about your health or diet..." autocomplete="off" />
        <button id="send-chat">
          <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
        </button>
      </div>
    </div>
    <button id="chat-toggle-btn" aria-label="Open AI Copilot">
      <svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z"/></svg>
    </button>
  `;
  document.body.appendChild(widgetContainer);

  const toggleBtn = document.getElementById("chat-toggle-btn");
  const closeBtn = document.getElementById("close-chat");
  const chatWindow = document.getElementById("chat-window");
  const chatBody = document.getElementById("chat-body");
  const chatInput = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-chat");
  const typingIndicator = document.getElementById("typing-indicator");

  let messages = [];

  function toggleChat() {
    chatWindow.classList.toggle("open");
    if (chatWindow.classList.contains("open")) {
      chatInput.focus();
    }
  }

  toggleBtn.addEventListener("click", toggleChat);
  closeBtn.addEventListener("click", toggleChat);

  // Hide custom landing page cursor when interacting with the chat widget
  widgetContainer.addEventListener("mouseenter", () => {
    const cursor = document.getElementById("cursor");
    const ring = document.getElementById("cursorRing");
    if (cursor) cursor.style.opacity = "0";
    if (ring) ring.style.opacity = "0";
  });
  widgetContainer.addEventListener("mouseleave", () => {
    const cursor = document.getElementById("cursor");
    const ring = document.getElementById("cursorRing");
    if (cursor) cursor.style.opacity = "1";
    if (ring) ring.style.opacity = "1";
  });

  function addMessage(text, sender) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `chat-message ${sender}`;
    msgDiv.textContent = text;
    chatBody.insertBefore(msgDiv, typingIndicator);
    chatBody.scrollTop = chatBody.scrollHeight;
  }

  function getContext() {
    let context = {};
    try {
      const riskStr = localStorage.getItem('athleteedge_risk_context');
      if (riskStr) context.latest_risk_scan = JSON.parse(riskStr);
    } catch(e){}
    try {
      const nutStr = localStorage.getItem('athleteedge_nutrition_context');
      if (nutStr) context.latest_nutrition_plan = JSON.parse(nutStr);
    } catch(e){}
    return context;
  }

  async function handleSend() {
    const text = chatInput.value.trim();
    if (!text) return;

    // Add user message
    addMessage(text, "user");
    messages.push({ role: "user", content: text });
    chatInput.value = "";
    
    // Show typing
    typingIndicator.classList.add("active");
    chatBody.scrollTop = chatBody.scrollHeight;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: messages,
          context: getContext()
        })
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Failed to fetch response");

      addMessage(data.reply, "bot");
      messages.push({ role: "assistant", content: data.reply });

    } catch (err) {
      addMessage("Sorry, I encountered an error: " + err.message, "bot");
    } finally {
      typingIndicator.classList.remove("active");
    }
  }

  sendBtn.addEventListener("click", handleSend);
  chatInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") handleSend();
  });
});
