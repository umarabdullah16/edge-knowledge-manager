import sys
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests  # REQUIRED: pip install requests

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QScrollArea, QFrame, QSplitter, 
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox, 
    QProgressBar, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QColor

# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = "http://localhost:8000"  # Address of your FastAPI server

# =============================================================================
# RAG BACKEND - Handles API Communication
# =============================================================================

class RAGBackend:
    """Interacts with the FastAPI backend for RAG operations"""
    
    def process_query(self, query: str) -> str:
        """Sends query to FastAPI /query endpoint"""
        try:
            payload = {"query": query}
            response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("answer", "No answer received from server.")
            else:
                error_detail = response.json().get('detail', response.text)
                raise Exception(f"Server Error ({response.status_code}): {error_detail}")
                
        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to the backend server. Is api.py running?")
        except Exception as e:
            raise Exception(f"Error: {str(e)}")

    def upload_document(self, file_path: str) -> dict:
        """Uploads a PDF to the FastAPI /ingest endpoint"""
        if not os.path.exists(file_path):
            raise Exception("File does not exist")
            
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
                response = requests.post(f"{API_BASE_URL}/ingest", files=files, timeout=120)
                
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get('detail', response.text)
                raise Exception(f"Upload Failed ({response.status_code}): {error_detail}")
                
        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to the backend server. Is api.py running?")
        except Exception as e:
            raise Exception(f"Error: {str(e)}")

# =============================================================================
# BACKGROUND THREADS
# =============================================================================

class QueryThread(QThread):
    """Runs the query request in background"""
    finished = Signal(str)  # Emits the answer
    error = Signal(str)     # Emits error message
    
    def __init__(self, query: str):
        super().__init__()
        self.query = query
        self.backend = RAGBackend()
        
    def run(self):
        try:
            response = self.backend.process_query(self.query)
            self.finished.emit(response)
        except Exception as e:
            self.error.emit(str(e))

class IngestionThread(QThread):
    """Runs the file upload/ingestion in background"""
    finished = Signal(str)  # Emits success message
    error = Signal(str)     # Emits error message
    
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.backend = RAGBackend()
        
    def run(self):
        try:
            result = self.backend.upload_document(self.file_path)
            # Format a nice success message
            msg = f"Successfully ingested {result.get('filename')}.\nProcessed {result.get('chunks_processed')} chunks."
            self.finished.emit(msg)
        except Exception as e:
            self.error.emit(str(e))

# =============================================================================
# LOADING ANIMATION
# =============================================================================

class LoadingAnimation(QWidget):
    """Creates the elegant purple progress bar that pulses while AI is thinking"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.pulse_effect)
        
        self.pulse_opacity = 1.0
        self.pulse_direction = -1
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setObjectName("elegantProgress")
        
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)
        self.setFixedSize(200, 20)
        
    def start_animation(self):
        self.timer.start(50)
        self.show()
        
    def stop_animation(self):
        self.timer.stop()
        self.hide()
        
    def pulse_effect(self):
        self.pulse_opacity += self.pulse_direction * 0.02
        if self.pulse_opacity <= 0.6:
            self.pulse_direction = 1
        elif self.pulse_opacity >= 1.0:
            self.pulse_direction = -1
            
        self.progress_bar.setStyleSheet(f"""
            QProgressBar#elegantProgress {{
                background-color: rgba(58, 58, 58, 0.8);
                border-radius: 3px;
                border: none;
            }}
            QProgressBar#elegantProgress::chunk {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 rgba(221, 160, 255, {self.pulse_opacity}),
                    stop: 0.5 rgba(187, 134, 252, {self.pulse_opacity}),
                    stop: 1 rgba(165, 112, 247, {self.pulse_opacity}));
                border-radius: 3px;
            }}
        """)

# =============================================================================
# UI COMPONENTS (Bubbles, Conversation, etc.)
# =============================================================================

class MessageBubble(QFrame):
    def __init__(self, message: str, is_user: bool = False, timestamp: str = None):
        super().__init__()
        self.is_user = is_user
        self.setup_ui(message, timestamp)
        self.add_shadow_effect()
        
    def setup_ui(self, message: str, timestamp: str):
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)
        
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setFont(QFont("Segoe UI", 12))
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse) # Allow copying text
        
        if self.is_user:
            self.setObjectName("userMessage")
            message_label.setStyleSheet("color: white; line-height: 1.4;")
        else:
            self.setObjectName("aiMessage")
            message_label.setStyleSheet("color: #e8e8e8; line-height: 1.4;")
            
        layout.addWidget(message_label)
        
        if timestamp:
            time_label = QLabel(timestamp)
            time_label.setFont(QFont("Segoe UI", 9))
            time_label.setStyleSheet("color: rgba(200, 200, 200, 0.7);")
            layout.addWidget(time_label)
            
        self.setLayout(layout)
        
    def add_shadow_effect(self):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 25))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

class Conversation:
    def __init__(self, title: str = "New Conversation"):
        self.id = str(int(time.time() * 1000))
        self.title = title
        self.messages: List[Dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()
        self.last_message_time = datetime.now()
        
    def add_message(self, content: str, is_user: bool, timestamp: str = None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M")
        
        self.messages.append({
            "content": content,
            "is_user": is_user,
            "timestamp": timestamp
        })
        self.last_message_time = datetime.now()
        
        if is_user and len([m for m in self.messages if m["is_user"]]) == 1:
            clean_title = content.strip()
            self.title = clean_title[:35] + "..." if len(clean_title) > 35 else clean_title
    
    def get_relative_time(self) -> str:
        now = datetime.now()
        diff = now - self.last_message_time
        if diff.seconds < 60: return "Just now"
        elif diff.seconds < 3600: return f"{diff.seconds // 60}m ago"
        elif diff.days == 0: return f"{diff.seconds // 3600}h ago"
        elif diff.days == 1: return "Yesterday"
        else: return f"{diff.days}d ago"
    
    def to_dict(self):
        return {
            "id": self.id, "title": self.title, "messages": self.messages,
            "created_at": self.created_at, "last_message_time": self.last_message_time.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        conv = cls(data["title"])
        conv.id = data["id"]
        conv.messages = data["messages"]
        conv.created_at = data["created_at"]
        if "last_message_time" in data:
            conv.last_message_time = datetime.fromisoformat(data["last_message_time"])
        return conv

class ConversationItem(QWidget):
    def __init__(self, conversation: Conversation):
        super().__init__()
        self.conversation = conversation
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        
        title_label = QLabel(self.conversation.title)
        title_label.setFont(QFont("Segoe UI", 12, QFont.Medium))
        title_label.setStyleSheet("color: #e8e8e8;")
        layout.addWidget(title_label)
        
        time_label = QLabel(self.conversation.get_relative_time())
        time_label.setFont(QFont("Segoe UI", 9))
        time_label.setStyleSheet("color: rgba(200, 200, 200, 0.6);")
        layout.addWidget(time_label)
        
        self.setLayout(layout)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

class RAGAssistant(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.conversations: List[Conversation] = []
        self.current_conversation: Optional[Conversation] = None
        self.data_file = "conversations.json"
        
        # Threads
        self.query_thread: Optional[QueryThread] = None
        self.ingest_thread: Optional[IngestionThread] = None
        
        self.setup_ui()
        self.load_conversations()
        self.apply_styles()
        
    def apply_styles(self):
        style_sheet = """
        /* Global */
        * { font-family: "Segoe UI", "Arial", sans-serif; }
        QMainWindow { background-color: #1a1a1a; color: #ffffff; }

        /* Sidebar */
        #sidebar { background-color: #242424; border-right: 1px solid #3a3a3a; min-width: 300px; max-width: 300px; }
        
        #newChatButton { background-color: #bb86fc; color: white; border: none; border-radius: 8px; font-size: 13px; font-weight: 600; padding: 12px 16px; }
        #newChatButton:hover { background-color: #c79bff; }
        #newChatButton:pressed { background-color: #a570f7; }

        #conversationsList { background-color: transparent; border: none; outline: none; }
        #conversationsList::item { background-color: transparent; border-radius: 8px; margin: 3px 0px; padding: 0px; }
        #conversationsList::item:hover { background-color: rgba(187, 134, 252, 0.1); }
        #conversationsList::item:selected { background-color: rgba(187, 134, 252, 0.15); border-left: 3px solid #bb86fc; }

        /* Chat Area */
        #chatArea { background-color: #1a1a1a; }
        #chatHeader { background-color: #242424; border-bottom: 1px solid #3a3a3a; }

        #uploadButton { background-color: #2d2d2d; color: #e8e8e8; border: 1px solid #4a4a4a; border-radius: 8px; font-size: 12px; font-weight: 500; padding: 8px 16px; }
        #uploadButton:hover { background-color: #3a3a3a; border-color: #bb86fc; color: #ffffff; }
        #uploadButton:pressed { background-color: #4a4a4a; }

        #messagesArea { background-color: #1a1a1a; border: none; }
        
        #userMessage { background-color: #bb86fc; border-radius: 16px; margin: 3px 80px 8px 0px; }
        #aiMessage { background-color: #2d2d2d; border-radius: 16px; margin: 3px 0px 8px 80px; border: 1px solid #3a3a3a; }

        #loadingContainer { background-color: rgba(36, 36, 36, 0.9); border: 1px solid rgba(187, 134, 252, 0.2); border-radius: 12px; margin: 0px 24px; }

        /* Input Area */
        #inputArea { background-color: #242424; border-top: 1px solid #3a3a3a; }
        #inputField { background-color: #1a1a1a; border: 2px solid #3a3a3a; border-radius: 12px; padding: 12px 16px; color: #ffffff; font-size: 13px; selection-background-color: #bb86fc; selection-color: white; }
        #inputField:focus { border-color: #bb86fc; outline: none; background-color: #1e1e1e; }
        #inputField::placeholder { color: #888888; }

        #sendButton { background-color: #bb86fc; color: white; border: none; border-radius: 10px; font-size: 13px; font-weight: 600; padding: 12px 20px; }
        #sendButton:hover { background-color: #c79bff; }
        #sendButton:pressed { background-color: #a570f7; }
        #sendButton:disabled { background-color: #4a4a4a; color: #888888; }

        /* Scrollbars */
        QScrollBar:vertical { background-color: #242424; width: 6px; border-radius: 3px; }
        QScrollBar::handle:vertical { background-color: #4a4a4a; border-radius: 3px; min-height: 20px; }
        QScrollBar::handle:vertical:hover { background-color: #bb86fc; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { border: none; background: none; height: 0px; }

        /* Dialogs */
        QMessageBox { background-color: #242424; color: #ffffff; border: 1px solid #3a3a3a; }
        QMessageBox QPushButton { background-color: #bb86fc; color: white; border: none; border-radius: 6px; padding: 8px 20px; }
        QFileDialog { background-color: #242424; color: #ffffff; }
        """
        self.setStyleSheet(style_sheet)
        
    def setup_ui(self):
        self.setWindowTitle("RAG Assistant")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        central_widget.setLayout(QHBoxLayout())
        central_widget.layout().addWidget(splitter)
        central_widget.layout().setContentsMargins(0, 0, 0, 0)
        
        self.setup_sidebar(splitter)
        self.setup_chat_area(splitter)
        
        splitter.setSizes([300, 900])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        
    def setup_sidebar(self, parent):
        sidebar = QWidget()
        sidebar.setFixedWidth(300)
        sidebar.setObjectName("sidebar")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        header_widget = QWidget()
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        
        title_label = QLabel("RAG Assistant")
        title_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title_label.setStyleSheet("color: #ffffff;")
        header_layout.addWidget(title_label)
        
        subtitle_label = QLabel("AI Document Analysis")
        subtitle_label.setFont(QFont("Segoe UI", 11))
        subtitle_label.setStyleSheet("color: #b0b0b0;")
        header_layout.addWidget(subtitle_label)
        
        header_widget.setLayout(header_layout)
        layout.addWidget(header_widget)
        
        self.new_chat_btn = QPushButton("✨ New Conversation")
        self.new_chat_btn.setFixedHeight(42)
        self.new_chat_btn.setObjectName("newChatButton")
        self.new_chat_btn.clicked.connect(self.create_new_conversation)
        layout.addWidget(self.new_chat_btn)
        
        self.conversations_list = QListWidget()
        self.conversations_list.setObjectName("conversationsList")
        self.conversations_list.itemClicked.connect(self.select_conversation)
        layout.addWidget(self.conversations_list)
        
        sidebar.setLayout(layout)
        parent.addWidget(sidebar)
        
    def setup_chat_area(self, parent):
        chat_widget = QWidget()
        chat_widget.setObjectName("chatArea")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        header = QWidget()
        header.setFixedHeight(60)
        header.setObjectName("chatHeader")
        
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(24, 0, 24, 0)
        
        status_layout = QHBoxLayout()
        status_layout.setSpacing(8)
        
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #4CAF50; font-size: 12px;")
        status_layout.addWidget(self.status_dot)
        
        self.chat_title = QLabel("Select or create a conversation")
        self.chat_title.setFont(QFont("Segoe UI", 15, QFont.Medium))
        self.chat_title.setStyleSheet("color: #ffffff; border-bottom: 1px solid rgba(187, 134, 252, 0.3);")
        status_layout.addWidget(self.chat_title)
        
        header_layout.addLayout(status_layout)
        header_layout.addStretch()
        
        self.upload_btn = QPushButton("📄  Upload Document")
        self.upload_btn.setFixedSize(140, 36)
        self.upload_btn.setObjectName("uploadButton")
        self.upload_btn.clicked.connect(self.upload_file)
        header_layout.addWidget(self.upload_btn)
        
        header.setLayout(header_layout)
        layout.addWidget(header)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setObjectName("messagesArea")
        
        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout()
        self.messages_layout.setContentsMargins(24, 24, 24, 24)
        self.messages_layout.setSpacing(16)
        self.messages_layout.addStretch()
        self.messages_widget.setLayout(self.messages_layout)
        
        scroll_area.setWidget(self.messages_widget)
        layout.addWidget(scroll_area)
        
        self.loading_widget = QWidget()
        self.loading_widget.setFixedHeight(60)
        self.loading_widget.hide()
        self.loading_widget.setObjectName("loadingContainer")
        
        loading_layout = QVBoxLayout()
        loading_layout.setContentsMargins(24, 15, 24, 15)
        loading_layout.setSpacing(12)
        
        self.loading_label = QLabel("AI is processing your request...")
        self.loading_label.setFont(QFont("Segoe UI", 12, QFont.Medium))
        self.loading_label.setStyleSheet("color: #dda0ff; font-weight: 500;")
        self.loading_label.setAlignment(Qt.AlignCenter)
        loading_layout.addWidget(self.loading_label)
        
        self.loading_animation = LoadingAnimation()
        loading_layout.addWidget(self.loading_animation, 0, Qt.AlignCenter)
        
        self.loading_widget.setLayout(loading_layout)
        layout.addWidget(self.loading_widget)
        
        input_area = QWidget()
        input_area.setFixedHeight(90)
        input_area.setObjectName("inputArea")
        
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(24, 16, 24, 16)
        input_layout.setSpacing(16)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask about your documents...")
        self.input_field.setFont(QFont("Segoe UI", 13))
        self.input_field.returnPressed.connect(self.send_message)
        self.input_field.setObjectName("inputField")
        input_layout.addWidget(self.input_field)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.setFixedSize(90, 42)
        self.send_btn.setObjectName("sendButton")
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)
        
        input_area.setLayout(input_layout)
        layout.addWidget(input_area)
        
        chat_widget.setLayout(layout)
        parent.addWidget(chat_widget)
    
    def create_new_conversation(self):
        conversation = Conversation()
        self.conversations.insert(0, conversation)
        self.current_conversation = conversation
        self.update_conversations_list()
        self.update_chat_display()
        self.save_conversations()
        self.input_field.setFocus()
        
    def update_conversations_list(self):
        self.conversations_list.clear()
        for conv in self.conversations:
            item = QListWidgetItem()
            item_widget = ConversationItem(conv)
            item.setSizeHint(item_widget.sizeHint())
            self.conversations_list.addItem(item)
            self.conversations_list.setItemWidget(item, item_widget)
            
        if self.current_conversation and self.conversations:
            try:
                index = self.conversations.index(self.current_conversation)
                self.conversations_list.setCurrentRow(index)
            except ValueError:
                pass
    
    def select_conversation(self, item):
        row = self.conversations_list.row(item)
        if 0 <= row < len(self.conversations):
            self.current_conversation = self.conversations[row]
            self.update_chat_display()
            
    def update_chat_display(self):
        for i in reversed(range(self.messages_layout.count() - 1)):
            item = self.messages_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
                
        if self.current_conversation:
            self.chat_title.setText(self.current_conversation.title)
            for msg in self.current_conversation.messages:
                self.add_message_bubble(msg["content"], msg["is_user"], msg.get("timestamp"))
        else:
            self.chat_title.setText("Select or create a conversation")
            
    def add_message_bubble(self, message: str, is_user: bool, timestamp: str = None):
        bubble = MessageBubble(message, is_user, timestamp)
        container = QWidget()
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        if is_user:
            container_layout.addStretch()
            container_layout.addWidget(bubble)
        else:
            container_layout.addWidget(bubble)
            container_layout.addStretch()
            
        container.setLayout(container_layout)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, container)
        QTimer.singleShot(100, self.scroll_to_bottom)
        
    def scroll_to_bottom(self):
        scroll_area = self.findChild(QScrollArea, "messagesArea")
        if scroll_area:
            scroll_bar = scroll_area.verticalScrollBar()
            scroll_bar.setValue(scroll_bar.maximum())
            
    def send_message(self):
        if not self.current_conversation:
            self.create_new_conversation()
            
        message = self.input_field.text().strip()
        if not message:
            return
            
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        
        timestamp = datetime.now().strftime("%H:%M")
        self.current_conversation.add_message(message, True, timestamp)
        self.add_message_bubble(message, True, timestamp)
        self.input_field.clear()
        
        self.show_loading("Thinking...")
        
        self.query_thread = QueryThread(message)
        self.query_thread.finished.connect(self.on_response_received)
        self.query_thread.error.connect(self.on_error_received)
        self.query_thread.start()
        
        self.update_conversations_list()
        self.save_conversations()
        
    def upload_file(self):
        # We only accept PDF since the backend assumes it
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Upload Document", "",
            "PDF Documents (*.pdf);;All Files (*)"
        )
        
        if file_path:
            self.show_loading(f"Ingesting {os.path.basename(file_path)}...")
            
            # Disable upload button while processing
            self.upload_btn.setEnabled(False)
            
            self.ingest_thread = IngestionThread(file_path)
            self.ingest_thread.finished.connect(self.on_ingestion_finished)
            self.ingest_thread.error.connect(self.on_ingestion_error)
            self.ingest_thread.start()

    def on_ingestion_finished(self, msg: str):
        self.hide_loading()
        self.upload_btn.setEnabled(True)
        QMessageBox.information(self, "Upload Complete", msg)
        
    def on_ingestion_error(self, error: str):
        self.hide_loading()
        self.upload_btn.setEnabled(True)
        QMessageBox.critical(self, "Upload Failed", f"Could not ingest document:\n{error}")

    def show_loading(self, message: str = "AI is processing..."):
        self.loading_label.setText(message)
        self.loading_widget.show()
        self.loading_animation.start_animation()
        
    def hide_loading(self):
        self.loading_widget.hide()
        self.loading_animation.stop_animation()
        
    def on_response_received(self, response: str):
        self.hide_loading()
        timestamp = datetime.now().strftime("%H:%M")
        self.current_conversation.add_message(response, False, timestamp)
        self.add_message_bubble(response, False, timestamp)
        
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.input_field.setFocus()
        
        self.update_conversations_list()
        self.save_conversations()
        
    def on_error_received(self, error: str):
        self.hide_loading()
        QMessageBox.warning(self, "Error", f"An error occurred: {error}")
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.input_field.setFocus()
            
    def save_conversations(self):
        try:
            data = [conv.to_dict() for conv in self.conversations]
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving conversations: {e}")
            
    def load_conversations(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.conversations = [Conversation.from_dict(conv_data) for conv_data in data]
                self.update_conversations_list()
        except Exception as e:
            print(f"Error loading conversations: {e}")
            self.conversations = []
            
    def closeEvent(self, event):
        self.save_conversations()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("RAG Assistant")
    app.setApplicationVersion("1.0")
    window = RAGAssistant()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()