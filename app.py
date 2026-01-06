import sys
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import random
from rag_backend import RAGBackend
from PySide6.QtCore import QThread, Signal
from typing import Optional
from PySide6.QtWidgets import QMessageBox
from rag_backend import RAGBackend




from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QScrollArea, QFrame, QSplitter, 
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox, 
    QProgressBar, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QColor

# =============================================================================
# LOADING ANIMATION - Beautiful progress bar with breathing effect
# =============================================================================

class LoadingAnimation(QWidget):
    """Creates the elegant purple progress bar that pulses while AI is thinking"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Timer for smooth animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.pulse_effect)
        
        # Animation state variables
        self.pulse_opacity = 1.0
        self.pulse_direction = -1  # -1 for dimming, 1 for brightening
        
    def setup_ui(self):
        """Create the progress bar widget"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Indeterminate progress bar (continuous animation)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 0,0 = indeterminate mode
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setObjectName("elegantProgress")
        
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)
        self.setFixedSize(200, 20)
        
    def start_animation(self):
        """Begin the breathing animation"""
        self.timer.start(50)  # Update every 50ms for smooth animation
        self.show()
        
    def stop_animation(self):
        """Stop animation and hide"""
        self.timer.stop()
        self.hide()
        
    def pulse_effect(self):
        """Creates the breathing/pulsing effect on progress bar"""
        # Gradually change opacity to create breathing effect
        self.pulse_opacity += self.pulse_direction * 0.02
        
        # Reverse direction when we hit limits
        if self.pulse_opacity <= 0.6:
            self.pulse_direction = 1    # Start brightening
        elif self.pulse_opacity >= 1.0:
            self.pulse_direction = -1   # Start dimming
            
        # Apply the new opacity to create beautiful gradient
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
# MESSAGE BUBBLES - Individual chat messages with styling
# =============================================================================

class MessageBubble(QFrame):
    """Creates individual message bubbles for user and AI messages"""
    
    def __init__(self, message: str, is_user: bool = False, timestamp: str = None):
        super().__init__()
        self.is_user = is_user
        self.setup_ui(message, timestamp)
        self.add_shadow_effect()  # Add subtle shadow for depth
        
    def setup_ui(self, message: str, timestamp: str):
        """Build the message bubble layout"""
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 12, 16, 12)  # Padding inside bubble
        layout.setSpacing(6)
        
        # Main message text
        message_label = QLabel(message)
        message_label.setWordWrap(True)  # Allow text to wrap to multiple lines
        message_label.setFont(QFont("Segoe UI", 12))
        message_label.setStyleSheet("line-height: 1.4;")  # Better line spacing
        
        # Different colors for user vs AI messages
        if self.is_user:
            self.setObjectName("userMessage")  # CSS will make this purple
            message_label.setStyleSheet("color: white; line-height: 1.4;")
        else:
            self.setObjectName("aiMessage")    # CSS will make this dark gray
            message_label.setStyleSheet("color: #e8e8e8; line-height: 1.4;")
            
        layout.addWidget(message_label)
        
        # Add timestamp if provided
        if timestamp:
            time_label = QLabel(timestamp)
            time_label.setFont(QFont("Segoe UI", 9))
            time_label.setStyleSheet("color: rgba(200, 200, 200, 0.7);")  # Subtle gray
            layout.addWidget(time_label)
            
        self.setLayout(layout)
        
    def add_shadow_effect(self):
        """Add subtle drop shadow to make bubbles look elevated"""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 25))  # Very subtle black shadow
        shadow.setOffset(0, 2)  # Shadow slightly below bubble
        self.setGraphicsEffect(shadow)

# =============================================================================
# CONVERSATION MANAGEMENT - Handles chat history and persistence
# =============================================================================

class Conversation:
    """Represents a single chat conversation with all its messages"""
    
    def __init__(self, title: str = "New Conversation"):
        # Unique ID based on timestamp
        self.id = str(int(time.time() * 1000))
        self.title = title
        self.messages: List[Dict[str, Any]] = []  # List of all messages
        self.created_at = datetime.now().isoformat()
        self.last_message_time = datetime.now()
        
    def add_message(self, content: str, is_user: bool, timestamp: str = None):
        """Add a new message to this conversation"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M")
        
        # Store message data
        self.messages.append({
            "content": content,
            "is_user": is_user,
            "timestamp": timestamp
        })
        
        self.last_message_time = datetime.now()
        
        # Auto-generate conversation title from first user message
        if is_user and len([m for m in self.messages if m["is_user"]]) == 1:
            clean_title = content.strip()
            # Truncate long titles
            self.title = clean_title[:35] + "..." if len(clean_title) > 35 else clean_title
    
    def get_relative_time(self) -> str:
        """Get human-friendly time like '2m ago' or 'Yesterday'"""
        now = datetime.now()
        diff = now - self.last_message_time
        
        if diff.seconds < 60:
            return "Just now"
        elif diff.seconds < 3600:
            return f"{diff.seconds // 60}m ago"
        elif diff.days == 0:
            return f"{diff.seconds // 3600}h ago"
        elif diff.days == 1:
            return "Yesterday"
        else:
            return f"{diff.days}d ago"
    
    def to_dict(self):
        """Convert to dictionary for JSON storage"""
        return {
            "id": self.id,
            "title": self.title,
            "messages": self.messages,
            "created_at": self.created_at,
            "last_message_time": self.last_message_time.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create conversation from stored JSON data"""
        conv = cls(data["title"])
        conv.id = data["id"]
        conv.messages = data["messages"]
        conv.created_at = data["created_at"]
        if "last_message_time" in data:
            conv.last_message_time = datetime.fromisoformat(data["last_message_time"])
        return conv

# =============================================================================
# CONVERSATION LIST ITEMS - Custom widgets for sidebar
# =============================================================================

class ConversationItem(QWidget):
    """Custom widget for displaying conversation in sidebar list"""
    
    def __init__(self, conversation: Conversation):
        super().__init__()
        self.conversation = conversation
        self.setup_ui()
        
    def setup_ui(self):
        """Create the conversation list item layout"""
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        
        # Conversation title
        title_label = QLabel(self.conversation.title)
        title_label.setFont(QFont("Segoe UI", 12, QFont.Medium))
        title_label.setStyleSheet("color: #e8e8e8;")
        layout.addWidget(title_label)
        
        # Last activity time ("2m ago", etc)
        time_label = QLabel(self.conversation.get_relative_time())
        time_label.setFont(QFont("Segoe UI", 9))
        time_label.setStyleSheet("color: rgba(200, 200, 200, 0.6);")
        layout.addWidget(time_label)
        
        self.setLayout(layout)

# =============================================================================
# MAIN APPLICATION - Core UI and application logic
# =============================================================================

class RAGAssistant(QMainWindow):
    """Main application window - handles all UI and user interactions"""
    
    def __init__(self):
        super().__init__()

        self.backend = RAGBackend()
        
        # App state variables
        self.conversations: List[Conversation] = []
        self.current_conversation: Optional[Conversation] = None
        self.data_file = "conversations.json"  # Where we save chat history
        self.rag_thread: Optional[RAGThread] = None  # Background processing thread
        
        # Initialize the application
        self.setup_ui()           # Build the interface
        self.load_conversations() # Load saved chats
        self.apply_styles()       # Apply purple theme
        
    def apply_styles(self):
        """Apply the purple dark theme CSS - no transform properties"""
        style_sheet = """
        /* RAG Assistant - Professional Dark Theme with Purple Accents */
        
        /* Global font for everything */
        * {
            font-family: "Segoe UI", "Arial", sans-serif;
        }

        /* Main window background */
        QMainWindow {
            background-color: #1a1a1a;
            color: #ffffff;
        }

        /* Left sidebar styling */
        #sidebar {
            background-color: #242424;
            border-right: 1px solid #3a3a3a;
            min-width: 300px;
            max-width: 300px;
        }

        /* New conversation button */
        #newChatButton {
            background-color: #bb86fc;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            padding: 12px 16px;
        }

        #newChatButton:hover {
            background-color: #c79bff;
        }

        #newChatButton:pressed {
            background-color: #a570f7;
        }

        /* Conversation list styling */
        #conversationsList {
            background-color: transparent;
            border: none;
            outline: none;
            selection-background-color: transparent;
        }

        #conversationsList::item {
            background-color: transparent;
            border-radius: 8px;
            margin: 3px 0px;
            padding: 0px;
        }

        #conversationsList::item:hover {
            background-color: rgba(187, 134, 252, 0.1);
        }

        #conversationsList::item:selected {
            background-color: rgba(187, 134, 252, 0.15);
            border-left: 3px solid #bb86fc;
        }

        /* Main chat area */
        #chatArea {
            background-color: #1a1a1a;
        }

        /* Chat header bar */
        #chatHeader {
            background-color: #242424;
            border-bottom: 1px solid #3a3a3a;
        }

        /* File upload button */
        #uploadButton {
            background-color: #2d2d2d;
            color: #e8e8e8;
            border: 1px solid #4a4a4a;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 500;
            padding: 8px 16px;
        }

        #uploadButton:hover {
            background-color: #3a3a3a;
            border-color: #bb86fc;
            color: #ffffff;
        }

        #uploadButton:pressed {
            background-color: #4a4a4a;
        }

        /* Messages scroll area */
        #messagesArea {
            background-color: #1a1a1a;
            border: none;
        }

        /* Custom scrollbar for messages */
        #messagesArea QScrollBar:vertical {
            background-color: #242424;
            width: 6px;
            border-radius: 3px;
            margin: 0px;
        }

        #messagesArea QScrollBar::handle:vertical {
            background-color: #4a4a4a;
            border-radius: 3px;
            min-height: 30px;
        }

        #messagesArea QScrollBar::handle:vertical:hover {
            background-color: #bb86fc;
        }

        #messagesArea QScrollBar::add-line:vertical,
        #messagesArea QScrollBar::sub-line:vertical {
            border: none;
            background: none;
            height: 0px;
        }

        /* User message bubbles (purple) */
        #userMessage {
            background-color: #bb86fc;
            border-radius: 16px;
            margin: 3px 80px 8px 0px;
        }

        /* AI message bubbles (dark gray) */
        #aiMessage {
            background-color: #2d2d2d;
            border-radius: 16px;
            margin: 3px 0px 8px 80px;
            border: 1px solid #3a3a3a;
        }

        /* Loading animation container */
        #loadingContainer {
            background-color: rgba(36, 36, 36, 0.9);
            border: 1px solid rgba(187, 134, 252, 0.2);
            border-radius: 12px;
            margin: 0px 24px;
        }

        /* Bottom input area */
        #inputArea {
            background-color: #242424;
            border-top: 1px solid #3a3a3a;
        }

        /* Text input field */
        #inputField {
            background-color: #1a1a1a;
            border: 2px solid #3a3a3a;
            border-radius: 12px;
            padding: 12px 16px;
            color: #ffffff;
            font-size: 13px;
            selection-background-color: #bb86fc;
            selection-color: white;
        }

        #inputField:focus {
            border-color: #bb86fc;
            outline: none;
            background-color: #1e1e1e;
        }

        #inputField::placeholder {
            color: #888888;
        }

        /* Send button */
        #sendButton {
            background-color: #bb86fc;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 600;
            padding: 12px 20px;
        }

        #sendButton:hover {
            background-color: #c79bff;
        }

        #sendButton:pressed {
            background-color: #a570f7;
        }

        #sendButton:disabled {
            background-color: #4a4a4a;
            color: #888888;
        }

        /* Splitter between sidebar and chat */
        QSplitter::handle {
            background-color: #3a3a3a;
            width: 1px;
        }

        QSplitter::handle:hover {
            background-color: #bb86fc;
        }

        /* Sidebar list scrollbar */
        QListWidget QScrollBar:vertical {
            background-color: #242424;
            width: 6px;
            border-radius: 3px;
            margin: 0px;
        }

        QListWidget QScrollBar::handle:vertical {
            background-color: #4a4a4a;
            border-radius: 3px;
            min-height: 20px;
        }

        QListWidget QScrollBar::handle:vertical:hover {
            background-color: #bb86fc;
        }

        QListWidget QScrollBar::add-line:vertical,
        QListWidget QScrollBar::sub-line:vertical {
            border: none;
            background: none;
            height: 0px;
        }

        /* Dialog boxes */
        QMessageBox {
            background-color: #242424;
            color: #ffffff;
            border: 1px solid #3a3a3a;
            border-radius: 8px;
        }

        QMessageBox QPushButton {
            background-color: #bb86fc;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 20px;
            font-size: 12px;
            font-weight: 500;
            min-width: 80px;
        }

        QMessageBox QPushButton:hover {
            background-color: #c79bff;
        }

        QMessageBox QPushButton:pressed {
            background-color: #a570f7;
        }

        /* File picker dialog */
        QFileDialog {
            background-color: #242424;
            color: #ffffff;
        }

        QFileDialog QPushButton {
            background-color: #bb86fc;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 16px;
            font-size: 11px;
        }

        QFileDialog QPushButton:hover {
            background-color: #c79bff;
        }

        /* Tooltips */
        QToolTip {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 1px solid #4a4a4a;
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 11px;
        }
        """
        
        self.setStyleSheet(style_sheet)
        
    def setup_ui(self):
        """Build the main application interface"""
        # Basic window setup
        self.setWindowTitle("RAG Assistant")
        self.setGeometry(100, 100, 1200, 800)  # x, y, width, height
        self.setMinimumSize(1000, 600)  # Don't allow window to be too small
        
        # Create central widget (everything goes inside this)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create horizontal splitter (divides sidebar from chat area)
        splitter = QSplitter(Qt.Horizontal)
        central_widget.setLayout(QHBoxLayout())
        central_widget.layout().addWidget(splitter)
        central_widget.layout().setContentsMargins(0, 0, 0, 0)
        
        # Build left sidebar and right chat area
        self.setup_sidebar(splitter)
        self.setup_chat_area(splitter)
        
        # Set initial proportions (300px sidebar, 900px chat)
        splitter.setSizes([300, 900])
        splitter.setCollapsible(0, False)  # Don't allow sidebar to collapse
        splitter.setCollapsible(1, False)  # Don't allow chat to collapse
        
    def setup_sidebar(self, parent):
        """Build the left sidebar with conversations list"""
        sidebar = QWidget()
        sidebar.setFixedWidth(300)
        sidebar.setObjectName("sidebar")  # For CSS styling
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)  # Padding around sidebar
        layout.setSpacing(20)
        
        # App title section
        header_widget = QWidget()
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        
        # Main title
        title_label = QLabel("RAG Assistant")
        title_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title_label.setStyleSheet("color: #ffffff;")
        header_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("AI Document Analysis")
        subtitle_label.setFont(QFont("Segoe UI", 11))
        subtitle_label.setStyleSheet("color: #b0b0b0;")
        header_layout.addWidget(subtitle_label)
        
        header_widget.setLayout(header_layout)
        layout.addWidget(header_widget)
        
        # New conversation button
        self.new_chat_btn = QPushButton("✨ New Conversation")
        self.new_chat_btn.setFixedHeight(42)
        self.new_chat_btn.setObjectName("newChatButton")
        self.new_chat_btn.clicked.connect(self.create_new_conversation)
        layout.addWidget(self.new_chat_btn)
        
        # List of saved conversations
        self.conversations_list = QListWidget()
        self.conversations_list.setObjectName("conversationsList")
        self.conversations_list.itemClicked.connect(self.select_conversation)
        layout.addWidget(self.conversations_list)
        
        sidebar.setLayout(layout)
        parent.addWidget(sidebar)
        
    def setup_chat_area(self, parent):
        """Build the main chat area on the right side"""
        chat_widget = QWidget()
        chat_widget.setObjectName("chatArea")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header bar with title and upload button
        header = QWidget()
        header.setFixedHeight(60)
        header.setObjectName("chatHeader")
        
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(24, 0, 24, 0)
        
        # Status indicator and title section
        status_layout = QHBoxLayout()
        status_layout.setSpacing(8)
        
        # Green dot showing connection status
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #4CAF50; font-size: 12px;")
        status_layout.addWidget(self.status_dot)
        
        # Chat title (shows current conversation name)
        self.chat_title = QLabel("Select or create a conversation")
        self.chat_title.setFont(QFont("Segoe UI", 15, QFont.Medium))
        self.chat_title.setStyleSheet("color: #ffffff; border-bottom: 1px solid rgba(187, 134, 252, 0.3);")
        status_layout.addWidget(self.chat_title)
        
        header_layout.addLayout(status_layout)
        header_layout.addStretch()  # Push upload button to right
        
        # File upload button
        self.upload_btn = QPushButton("📄  Upload Document")
        self.upload_btn.setFixedSize(140, 36)
        self.upload_btn.setObjectName("uploadButton")
        self.upload_btn.clicked.connect(self.upload_file)
        header_layout.addWidget(self.upload_btn)
        
        header.setLayout(header_layout)
        layout.addWidget(header)
        
        # Scrollable messages area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # No horizontal scroll
        scroll_area.setObjectName("messagesArea")
        
        # Container for all messages
        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout()
        self.messages_layout.setContentsMargins(24, 24, 24, 24)
        self.messages_layout.setSpacing(16)  # Space between messages
        self.messages_layout.addStretch()  # Push messages to top
        self.messages_widget.setLayout(self.messages_layout)
        
        scroll_area.setWidget(self.messages_widget)
        layout.addWidget(scroll_area)
        
        # Loading indicator (hidden by default)
        self.loading_widget = QWidget()
        self.loading_widget.setFixedHeight(60)
        self.loading_widget.hide()
        self.loading_widget.setObjectName("loadingContainer")
        
        loading_layout = QVBoxLayout()
        loading_layout.setContentsMargins(24, 15, 24, 15)
        loading_layout.setSpacing(12)
        
        # Loading text
        self.loading_label = QLabel("AI is processing your request...")
        self.loading_label.setFont(QFont("Segoe UI", 12, QFont.Medium))
        self.loading_label.setStyleSheet("color: #dda0ff; font-weight: 500;")
        self.loading_label.setAlignment(Qt.AlignCenter)
        loading_layout.addWidget(self.loading_label)
        
        # Progress bar animation
        self.loading_animation = LoadingAnimation()
        loading_layout.addWidget(self.loading_animation, 0, Qt.AlignCenter)
        
        self.loading_widget.setLayout(loading_layout)
        layout.addWidget(self.loading_widget)
        
        # Bottom input area
        input_area = QWidget()
        input_area.setFixedHeight(90)  # Slightly taller for better proportions
        input_area.setObjectName("inputArea")
        
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(24, 16, 24, 16)
        input_layout.setSpacing(16)
        
        # Text input field
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask about your documents...")
        self.input_field.setFont(QFont("Segoe UI", 13))
        self.input_field.returnPressed.connect(self.send_message)  # Enter key sends message
        self.input_field.setObjectName("inputField")
        input_layout.addWidget(self.input_field)
        
        # Send button
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
        """Create a new empty conversation"""
        conversation = Conversation()
        self.conversations.insert(0, conversation)  # Add to top of list
        self.current_conversation = conversation
        self.update_conversations_list()
        self.update_chat_display()
        self.save_conversations()
        self.input_field.setFocus()  # Focus on input field
        
    def update_conversations_list(self):
        """Refresh the sidebar conversation list"""
        self.conversations_list.clear()
        
        # Add each conversation as a custom widget
        for conv in self.conversations:
            item = QListWidgetItem()
            item_widget = ConversationItem(conv)
            item.setSizeHint(item_widget.sizeHint())
            self.conversations_list.addItem(item)
            self.conversations_list.setItemWidget(item, item_widget)
            
        # Highlight currently selected conversation
        if self.current_conversation and self.conversations:
            try:
                index = self.conversations.index(self.current_conversation)
                self.conversations_list.setCurrentRow(index)
            except ValueError:
                pass  # Conversation not found, no problem
    
    def select_conversation(self, item):
        """Handle clicking on a conversation in the sidebar"""
        row = self.conversations_list.row(item)
        if 0 <= row < len(self.conversations):
            self.current_conversation = self.conversations[row]
            self.update_chat_display()
            
    def update_chat_display(self):
        """Refresh the main chat area with current conversation messages"""
        # Clear existing message bubbles
        for i in reversed(range(self.messages_layout.count() - 1)):  # Keep the stretch at end
            item = self.messages_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
                
        if self.current_conversation:
            # Update title
            self.chat_title.setText(self.current_conversation.title)
            
            # Add all messages as bubbles
            for msg in self.current_conversation.messages:
                self.add_message_bubble(msg["content"], msg["is_user"], msg.get("timestamp"))
        else:
            # No conversation selected
            self.chat_title.setText("Select or create a conversation")
            
    def add_message_bubble(self, message: str, is_user: bool, timestamp: str = None):
        """Add a new message bubble to the chat display"""
        bubble = MessageBubble(message, is_user, timestamp)
        
        # Create container to control alignment
        container = QWidget()
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        if is_user:
            # User messages align right
            container_layout.addStretch()
            container_layout.addWidget(bubble)
        else:
            # AI messages align left
            container_layout.addWidget(bubble)
            container_layout.addStretch()
            
        container.setLayout(container_layout)
        
        # Insert before the stretch at the end
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, container)
        
        # Scroll to bottom after a short delay (allows widget to render)
        QTimer.singleShot(100, self.scroll_to_bottom)
        
    def scroll_to_bottom(self):
        """Auto-scroll to show the latest message"""
        scroll_area = self.findChild(QScrollArea, "messagesArea")
        if scroll_area:
            scroll_bar = scroll_area.verticalScrollBar()
            scroll_bar.setValue(scroll_bar.maximum())
            
    def send_message(self):
        """Handle sending a user message"""
        # Create conversation if none exists
        if not self.current_conversation:
            self.create_new_conversation()
            
        message = self.input_field.text().strip()
        if not message:
            return  # Don't send empty messages
            
        # Disable input while processing
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        
        # Add user message to conversation
        timestamp = datetime.now().strftime("%H:%M")
        self.current_conversation.add_message(message, True, timestamp)
        self.add_message_bubble(message, True, timestamp)
        
        # Clear input field
        self.input_field.clear()
        
        # Show loading animation
        self.show_loading()
        
        # Start background processing
        self.rag_thread = RAGThread(message)
        self.rag_thread.finished.connect(self.on_response_received)
        self.rag_thread.error.connect(self.on_error_received)
        self.rag_thread.start()
        
        # Update UI
        self.update_conversations_list()
        self.save_conversations()
        
    def show_loading(self):
        """Show the loading animation"""
        self.loading_widget.show()
        self.loading_animation.start_animation()
        
    def hide_loading(self):
        """Hide the loading animation"""
        self.loading_widget.hide()
        self.loading_animation.stop_animation()
        
    def on_response_received(self, response: str):
        """Handle AI response from background thread"""
        self.hide_loading()
        
        # Add AI response to conversation
        timestamp = datetime.now().strftime("%H:%M")
        self.current_conversation.add_message(response, False, timestamp)
        self.add_message_bubble(response, False, timestamp)
        
        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.input_field.setFocus()
        
        # Update and save
        self.update_conversations_list()
        self.save_conversations()
        
    def on_error_received(self, error: str):
        """Handle error from background processing"""
        self.hide_loading()
        QMessageBox.warning(self, "Error", f"An error occurred: {error}")
        
        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.input_field.setFocus()
        
    def upload_file(self):
        """Handle file upload button click"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Upload Document", "",
            "Documents (*.pdf *.docx *.txt *.doc *.md);;All Files (*)"
        )
        
        if file_path:
            try:
                self.backend.ingest_document(file_path)
                QMessageBox.information(
                    self,
                    "Success",
                    "Document ingested successfully."
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    str(e)
                )

            
    def save_conversations(self):
        """Save all conversations to JSON file"""
        try:
            data = [conv.to_dict() for conv in self.conversations]
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving conversations: {e}")
            
    def load_conversations(self):
        """Load conversations from JSON file on app startup"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Convert JSON data back to Conversation objects
                self.conversations = [Conversation.from_dict(conv_data) for conv_data in data]
                self.update_conversations_list()
        except Exception as e:
            print(f"Error loading conversations: {e}")
            self.conversations = []
            
    def closeEvent(self, event):
        """Save conversations when app is closed"""
        self.save_conversations()
        event.accept()

# =============================================================================
# BACKGROUND THREAD - Calls FastAPI backend
# =============================================================================

class RAGThread(QThread):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, query: str, file_path: Optional[str] = None):
        super().__init__()
        self.query = query
        self.file_path = file_path
        self.backend = RAGBackend()

    def run(self):
        try:
            response = self.backend.process_query(self.query, self.file_path)
            self.finished.emit(response)
        except Exception as e:
            self.error.emit(str(e))


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

def main():
    """Initialize and run the application"""
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("RAG Assistant")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = RAGAssistant()
    window.show()
    
    # Start the Qt event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
