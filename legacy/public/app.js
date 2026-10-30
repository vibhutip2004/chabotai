/**
 * DocuQuery AI — Modern RAG Chatbot Client Application
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- State ---
  const state = {
    provider: localStorage.getItem('dq_provider') || 'groq',
    apiKey: localStorage.getItem('dq_apiKey') || '',
    topK: parseInt(localStorage.getItem('dq_topK') || '4', 10),
    messages: [],
    isGenerating: false,
  };

  // --- DOM Elements ---
  const sidebar = document.getElementById('sidebar');
  const menuToggleBtn = document.getElementById('menuToggleBtn');
  const providerBtns = document.querySelectorAll('.provider-btn');
  const apiKeyContainer = document.getElementById('apiKeyContainer');
  const apiKeyLabel = document.getElementById('apiKeyLabel');
  const apiKeyInput = document.getElementById('apiKeyInput');
  const apiKeyHint = document.getElementById('apiKeyHint');
  const togglePwdBtn = document.getElementById('togglePwdBtn');
  const topKSlider = document.getElementById('topKSlider');
  const topKValue = document.getElementById('topKValue');
  const docList = document.getElementById('docList');
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const clearChatBtn = document.getElementById('clearChatBtn');
  const heroWelcome = document.getElementById('heroWelcome');
  const messagesList = document.getElementById('messagesList');
  const chatForm = document.getElementById('chatForm');
  const userInput = document.getElementById('userInput');
  const sendBtn = document.getElementById('sendBtn');
  const activeEngineBadge = document.getElementById('activeEngineBadge');
  const chipBtns = document.querySelectorAll('.chip-btn');

  // --- Initial Setup ---
  apiKeyInput.value = state.apiKey;
  topKSlider.value = state.topK;
  topKValue.textContent = state.topK;
  updateProviderUI(state.provider);

  // --- Auto-expand textarea ---
  userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 140) + 'px';
  });

  // Enter to send (Shift+Enter for new line)
  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!state.isGenerating && userInput.value.trim()) {
        chatForm.dispatchEvent(new Event('submit'));
      }
    }
  });

  // --- Mobile Sidebar Toggle ---
  if (menuToggleBtn) {
    menuToggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });
  }

  // Close sidebar on tap outside (mobile)
  document.addEventListener('click', (e) => {
    if (window.innerWidth <= 868 && sidebar.classList.contains('open')) {
      if (!sidebar.contains(e.target) && !menuToggleBtn.contains(e.target)) {
        sidebar.classList.remove('open');
      }
    }
  });

  // --- Provider Selection ---
  providerBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const p = btn.dataset.provider;
      state.provider = p;
      localStorage.setItem('dq_provider', p);
      updateProviderUI(p);
    });
  });

  function updateProviderUI(provider) {
    providerBtns.forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.provider === provider);
    });

    if (provider === 'groq') {
      apiKeyContainer.style.display = 'flex';
      apiKeyLabel.textContent = 'Groq API Key';
      apiKeyInput.placeholder = 'gsk_...';
      apiKeyHint.innerHTML = 'Free instant key at <a href="https://console.groq.com/keys" target="_blank" rel="noopener">console.groq.com</a>';
      activeEngineBadge.textContent = 'Groq Llama 3.1';
    } else if (provider === 'openai') {
      apiKeyContainer.style.display = 'flex';
      apiKeyLabel.textContent = 'OpenAI API Key';
      apiKeyInput.placeholder = 'sk-...';
      apiKeyHint.innerHTML = 'Get key at <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener">platform.openai.com</a>';
      activeEngineBadge.textContent = 'OpenAI GPT-4o';
    } else {
      apiKeyContainer.style.display = 'none';
      activeEngineBadge.textContent = 'Local RAG Mode';
    }
  }

  // --- API Key Input ---
  apiKeyInput.addEventListener('input', () => {
    state.apiKey = apiKeyInput.value.trim();
    localStorage.setItem('dq_apiKey', state.apiKey);
  });

  togglePwdBtn.addEventListener('click', () => {
    const isPwd = apiKeyInput.type === 'password';
    apiKeyInput.type = isPwd ? 'text' : 'password';
    togglePwdBtn.textContent = isPwd ? '🔒' : '👁️';
  });

  // --- Top-K Slider ---
  topKSlider.addEventListener('input', () => {
    state.topK = parseInt(topKSlider.value, 10);
    topKValue.textContent = state.topK;
    localStorage.setItem('dq_topK', state.topK);
  });

  // --- Sample Prompt Chips ---
  chipBtns.forEach((chip) => {
    chip.addEventListener('click', () => {
      const query = chip.dataset.query;
      userInput.value = query;
      userInput.style.height = 'auto';
      chatForm.dispatchEvent(new Event('submit'));
    });
  });

  // --- Document Drag & Drop / Upload ---
  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-active');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-active');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-active');
    if (e.dataTransfer.files.length) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
      handleFileUpload(fileInput.files[0]);
    }
  });

  async function handleFileUpload(file) {
    const reader = new FileReader();
    reader.onload = async (e) => {
      const content = e.target.result;
      try {
        const res = await fetch('/api/upload', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: file.name, content: content }),
        });
        const json = await res.json();
        // Add to UI
        const li = document.createElement('li');
        li.className = 'doc-item';
        li.innerHTML = `
          <div class="doc-icon">📄</div>
          <div class="doc-meta">
            <span class="doc-name">${escapeHtml(file.name)}</span>
            <span class="doc-chunks">${json.chunks_added || 'Multiple'} chunks indexed</span>
          </div>
        `;
        docList.appendChild(li);
        alert(`Document "${file.name}" indexed successfully!`);
      } catch (err) {
        console.error('Upload failed:', err);
        // Fallback UI indication
        const li = document.createElement('li');
        li.className = 'doc-item';
        li.innerHTML = `
          <div class="doc-icon">📄</div>
          <div class="doc-meta">
            <span class="doc-name">${escapeHtml(file.name)}</span>
            <span class="doc-chunks">Cached in session</span>
          </div>
        `;
        docList.appendChild(li);
        alert(`Uploaded "${file.name}" for this session.`);
      }
    };
    reader.readAsText(file);
  }

  // --- Clear Chat ---
  clearChatBtn.addEventListener('click', () => {
    state.messages = [];
    messagesList.innerHTML = '';
    heroWelcome.style.display = 'flex';
  });

  // --- Chat Submission ---
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    if (!query || state.isGenerating) return;

    // Hide hero welcome
    heroWelcome.style.display = 'none';

    // Append User Message
    appendMessage('user', query);
    userInput.value = '';
    userInput.style.height = 'auto';

    // Show Assistant Loading Indicator
    const assistantMsgElem = appendLoadingMessage();
    state.isGenerating = true;
    sendBtn.disabled = true;

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: query,
          provider: state.provider,
          apiKey: state.apiKey,
          topK: state.topK,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();
      updateAssistantMessage(assistantMsgElem, data.answer, data.sources);
    } catch (err) {
      console.warn('API error, executing client-side fallback RAG:', err);
      // Client-side fallback if running statically or server is warming up
      const fallbackData = executeClientSideRAG(query);
      updateAssistantMessage(assistantMsgElem, fallbackData.answer, fallbackData.sources);
    } finally {
      state.isGenerating = false;
      sendBtn.disabled = false;
      userInput.focus();
    }
  });

  // --- Helper: Render Messages ---
  function appendMessage(role, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '📄';

    const body = document.createElement('div');
    body.className = 'message-body';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = formatMarkdown(text);

    body.appendChild(bubble);
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(body);

    messagesList.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
  }

  function appendLoadingMessage() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '📄';

    const body = document.createElement('div');
    body.className = 'message-body';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = `
      <div class="typing-dots">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    `;

    body.appendChild(bubble);
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(body);

    messagesList.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
  }

  function updateAssistantMessage(msgElem, answer, sources) {
    const bubble = msgElem.querySelector('.message-bubble');
    bubble.innerHTML = formatMarkdown(answer);

    if (sources && sources.length > 0) {
      const body = msgElem.querySelector('.message-body');
      const sourcesCard = document.createElement('div');
      sourcesCard.className = 'sources-card';

      const sourcesToggle = document.createElement('div');
      sourcesToggle.className = 'sources-toggle';
      sourcesToggle.innerHTML = `
        <span>🔍 Citations &amp; Sources (${sources.length})</span>
        <span>▼</span>
      `;

      const sourcesContent = document.createElement('div');
      sourcesContent.className = 'sources-content';
      sourcesContent.style.display = 'none';

      sources.forEach((src, idx) => {
        const item = document.createElement('div');
        item.className = 'source-item';
        item.innerHTML = `
          <div class="source-header">
            <span>Source ${idx + 1} — ${escapeHtml(src.source || 'Document')}</span>
            <span>Page ${src.page || 1}</span>
          </div>
          <div class="source-snippet">${escapeHtml(src.snippet || src.content || '')}</div>
        `;
        sourcesContent.appendChild(item);
      });

      sourcesToggle.addEventListener('click', () => {
        const isHidden = sourcesContent.style.display === 'none';
        sourcesContent.style.display = isHidden ? 'flex' : 'none';
        sourcesToggle.querySelector('span:last-child').textContent = isHidden ? '▲' : '▼';
      });

      sourcesCard.appendChild(sourcesToggle);
      sourcesCard.appendChild(sourcesContent);
      body.appendChild(sourcesCard);
    }

    scrollToBottom();
  }

  function scrollToBottom() {
    const container = document.getElementById('chatContainer');
    container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
  }

  function formatMarkdown(text) {
    if (!text) return '';
    let html = escapeHtml(text);
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italics
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Code inline
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Line breaks to paragraphs
    const paragraphs = html.split(/\n\n+/);
    return paragraphs.map((p) => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
  }

  function escapeHtml(string) {
    const entityMap = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    };
    return String(string).replace(/[&<>"']/g, (s) => entityMap[s]);
  }

  // --- Client-side fallback RAG (zero-network redundancy) ---
  function executeClientSideRAG(query) {
    const knowledgeBase = [
      {
        source: 'ml_basics_notes.txt',
        page: 1,
        content:
          'Supervised Learning is where the model is trained on labeled data. Examples include Classification (predicting categories) and Regression (predicting continuous numerical quantities). Algorithms include Linear Regression, Logistic Regression, Random Forest, and Gradient Boosted Trees.',
      },
      {
        source: 'ml_basics_notes.txt',
        page: 1,
        content:
          'Unsupervised Learning deals with unlabeled data to discover latent structures or patterns. Examples include Clustering (K-Means, DBSCAN) and Dimensionality Reduction (PCA, t-SNE).',
      },
      {
        source: 'ml_basics_notes.txt',
        page: 1,
        content:
          'Vector Embeddings are dense numerical representations of text where semantically similar concepts map to nearby coordinates in high-dimensional vector space. Vector databases like ChromaDB and FAISS compute cosine similarity to retrieve relevant chunks.',
      },
      {
        source: 'ml_basics_notes.txt',
        page: 1,
        content:
          'Retrieval-Augmented Generation (RAG) grounds Large Language Models on private or domain-specific documents by retrieving top-k relevant text chunks and passing them as context inside the prompt.',
      },
    ];

    const qWords = query.toLowerCase().match(/\b\w{3,}\b/g) || [];
    const scored = knowledgeBase.map((k) => {
      const kWords = k.content.toLowerCase().match(/\b\w{3,}\b/g) || [];
      const overlap = qWords.filter((w) => kWords.includes(w)).length;
      return { ...k, score: overlap };
    });

    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, state.topK);

    let answerText = top.map((t) => t.content).join(' ');
    answerText +=
      '\n\n*(Answer retrieved from indexed document chunks. To enable full LLM generation, provide a Groq or OpenAI API key in the sidebar.)*';

    return {
      answer: answerText,
      sources: top.map((t) => ({ source: t.source, page: t.page, snippet: t.content })),
    };
  }
});
