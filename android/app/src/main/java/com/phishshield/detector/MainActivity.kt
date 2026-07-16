package com.phishshield.detector

import android.Manifest
import android.content.Context
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.phishshield.detector.db.ScamDatabase
import com.phishshield.detector.db.ScamLog
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : ComponentActivity() {

    private lateinit var sharedPrefs: SharedPreferences
    private lateinit var database: ScamDatabase

    // Request permissions launcher
    private val requestPermissionsLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val receiveSmsGranted = permissions[Manifest.permission.RECEIVE_SMS] ?: false
        val readSmsGranted = permissions[Manifest.permission.READ_SMS] ?: false
        
        // Update SharedPreferences
        if (receiveSmsGranted && readSmsGranted) {
            sharedPrefs.edit().putBoolean("permissions_granted", true).apply()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        sharedPrefs = getSharedPreferences("phishshield_prefs", Context.MODE_PRIVATE)
        database = ScamDatabase.getDatabase(this)

        setContent {
            PhishShieldApp()
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun PhishShieldApp() {
        // App states
        var isProtectionEnabled by remember {
            mutableStateOf(sharedPrefs.getBoolean("sms_protection_enabled", true))
        }
        
        // Dynamic permission state
        var hasSmsPermissions by remember {
            mutableStateOf(hasRequiredPermissions())
        }

        // Onboarding consent state
        var consentGiven by remember {
            mutableStateOf(sharedPrefs.getBoolean("first_launch_consent_given", false))
        }

        // Observe local Review Logs flow
        val logsFlow = remember { database.scamLogDao().getAllLogs() }
        val logsList by logsFlow.collectAsState(initial = emptyList())
        val coroutineScope = rememberCoroutineScope()

        // Sync permission status
        LaunchedEffect(key1 = hasSmsPermissions) {
            hasSmsPermissions = hasRequiredPermissions()
        }

        MaterialTheme(
            colorScheme = darkColorScheme(
                primary = Color(0xFF8B5CF6), // Violet Accent
                secondary = Color(0xFFEC4899), // Pink Accent
                background = Color(0xFF0B0C10), // Deep Obsidian
                surface = Color(0xFF161A25), // Card Background
                error = Color(0xFFEF4444), // Neon Red
                onPrimary = Color.White,
                onSecondary = Color.White
            )
        ) {
            Surface(
                modifier = Modifier.fillMaxSize(),
                color = MaterialTheme.colorScheme.background
            ) {
                if (!consentGiven || !hasSmsPermissions) {
                    // Onboarding Consent & Onboarding Screen
                    ConsentScreen(
                        onConsentAccepted = {
                            sharedPrefs.edit().putBoolean("first_launch_consent_given", true).apply()
                            consentGiven = true
                            triggerPermissionRequest()
                        }
                    )
                } else {
                    // Main Dashboard View
                    Scaffold(
                        topBar = {
                            TopAppBar(
                                title = {
                                    Text(
                                        text = "PhishShield Protection",
                                        fontFamily = androidx.compose.ui.text.font.FontFamily.Default,
                                        fontWeight = FontWeight.Bold,
                                        color = Color.White
                                    )
                                },
                                colors = TopAppBarDefaults.topAppBarColors(
                                    containerColor = Color(0xFF0B0C10)
                                )
                            )
                        }
                    ) { paddingValues ->
                        Column(
                            modifier = Modifier
                                .fillMaxSize()
                                .padding(paddingValues)
                                .padding(16.dp)
                        ) {
                            // 1. Protection Status Card
                            StatusCard(
                                isEnabled = isProtectionEnabled,
                                onToggleChange = { enabled ->
                                    sharedPrefs.edit().putBoolean("sms_protection_enabled", enabled).apply()
                                    isProtectionEnabled = enabled
                                },
                                hasPermissions = hasSmsPermissions,
                                onRevoke = {
                                    // Set enabled state to false in prefs and trigger refresh
                                    sharedPrefs.edit().putBoolean("sms_protection_enabled", false).apply()
                                    isProtectionEnabled = false
                                }
                            )

                            Spacer(modifier = Modifier.height(24.dp))

                            // 2. Local Message Logs list header
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = "Flagged Message Log (${logsList.size})",
                                    fontSize = 16.sp,
                                    fontWeight = FontWeight.SemiBold,
                                    color = Color.White
                                )
                                if (logsList.isNotEmpty()) {
                                    TextButton(
                                        onClick = {
                                            coroutineScope.launch {
                                                database.scamLogDao().clearAll()
                                            }
                                        },
                                        colors = ButtonDefaults.textButtonColors(
                                            contentColor = MaterialTheme.colorScheme.error
                                        )
                                    ) {
                                        Icon(Icons.Default.Clear, contentDescription = "Clear All")
                                        Spacer(modifier = Modifier.width(4.dp))
                                        Text("Clear All")
                                    }
                                }
                            }

                            Spacer(modifier = Modifier.height(8.dp))

                            // 3. Log list
                            if (logsList.isEmpty()) {
                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .weight(1f)
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(MaterialTheme.colorScheme.surface)
                                        .padding(24.dp),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                        Icon(
                                            Icons.Default.Info,
                                            contentDescription = "Empty",
                                            tint = Color.Gray,
                                            modifier = Modifier.size(48.dp)
                                        )
                                        Spacer(modifier = Modifier.height(12.dp))
                                        Text(
                                            text = "No flagged messages detected locally.",
                                            color = Color.Gray,
                                            textAlign = TextAlign.Center
                                        )
                                    }
                                }
                            } else {
                                LazyColumn(
                                    modifier = Modifier.weight(1f),
                                    verticalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                    items(logsList) { log ->
                                        LogItemCard(
                                            log = log,
                                            onFeedbackSubmit = { id, feedback ->
                                                coroutineScope.launch {
                                                    database.scamLogDao().updateFeedback(id, feedback)
                                                }
                                            },
                                            onDelete = { id ->
                                                coroutineScope.launch {
                                                    database.scamLogDao().deleteLog(id)
                                                }
                                            }
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Composable
    fun ConsentScreen(onConsentAccepted: () -> Unit) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            verticalArrangement = Arrangement.SpaceBetween,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.Center
            ) {
                // Header Gradient Icon
                Box(
                    modifier = Modifier
                        .size(80.dp)
                        .clip(RoundedCornerShape(20.dp))
                        .background(
                            Brush.linearGradient(
                                colors = listOf(Color(0xFF8B5CF6), Color(0xFFEC4899))
                            )
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        Icons.Default.Warning,
                        contentDescription = "Shield Logo",
                        tint = Color.White,
                        modifier = Modifier.size(44.dp)
                    )
                }

                Spacer(modifier = Modifier.height(24.dp))

                Text(
                    text = "PhishShield Protection",
                    fontSize = 28.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Privacy-First SMS Phishing Detection",
                    fontSize = 14.sp,
                    color = Color(0xFFEC4899),
                    fontWeight = FontWeight.SemiBold
                )

                Spacer(modifier = Modifier.height(32.dp))

                // Onboarding Explanations Card
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surface
                    ),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Column(modifier = Modifier.padding(20.dp)) {
                        Text(
                            text = "PRIVACY & SECURITY GUARANTEES",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color.Gray,
                            letterSpacing = 1.sp
                        )
                        Spacer(modifier = Modifier.height(16.dp))

                        BulletItem("100% On-Device Analysis", "Your text messages are processed entirely locally in memory. Nothing leaves your phone.")
                        Spacer(modifier = Modifier.height(12.dp))
                        BulletItem("Zero Network Access", "This application does not request the Android Internet Permission. It is physically incapable of transmitting data.")
                        Spacer(modifier = Modifier.height(12.dp))
                        BulletItem("Sideloading Permissions", "SmsReceiver requires RECEIVE_SMS permissions to check message headers and alerts in real-time.")
                    }
                }
            }

            // Accept button
            Button(
                onClick = onConsentAccepted,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
                    .clip(RoundedCornerShape(12.dp)),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF8B5CF6)
                )
            ) {
                Text(
                    text = "Enable SMS Protection",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }

    @Composable
    fun BulletItem(title: String, desc: String) {
        Column {
            Text(text = "✓ $title", fontWeight = FontWeight.Bold, fontSize = 14.sp, color = Color(0xFF10B981))
            Spacer(modifier = Modifier.height(2.dp))
            Text(text = desc, fontSize = 13.sp, color = Color.LightGray, lineHeight = 18.sp)
        }
    }

    @Composable
    fun StatusCard(
        isEnabled: Boolean,
        onToggleChange: (Boolean) -> Unit,
        hasPermissions: Boolean,
        onRevoke: () -> Unit
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            ),
            shape = RoundedCornerShape(16.dp),
            border = BorderStroke(1.dp, Color(0xFF8B5CF6).copy(alpha = 0.2f))
        ) {
            Column(modifier = Modifier.padding(20.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = if (isEnabled && hasPermissions) "Active SMS Protection" else "SMS Protection Inactive",
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                        Text(
                            text = if (isEnabled && hasPermissions) "Monitoring incoming texts locally" else "No security receiver active",
                            fontSize = 12.sp,
                            color = Color.Gray
                        )
                    }

                    Switch(
                        checked = isEnabled && hasPermissions,
                        onCheckedChange = { onToggleChange(it) },
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = Color(0xFF10B981),
                            checkedTrackColor = Color(0xFF10B981).copy(alpha = 0.3f)
                        )
                    )
                }

                if (!hasPermissions) {
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = "Warning: App lacks required SMS runtime permissions to function.",
                        color = MaterialTheme.colorScheme.error,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))
                
                Divider(color = Color.White.copy(alpha = 0.05f))
                
                Spacer(modifier = Modifier.height(12.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Revoke Access Gates",
                        fontSize = 14.sp,
                        color = Color.LightGray
                    )
                    TextButton(
                        onClick = onRevoke,
                        colors = ButtonDefaults.textButtonColors(
                            contentColor = MaterialTheme.colorScheme.error
                        )
                    ) {
                        Text("Disable App Receiver")
                    }
                }
            }
        }
    }

    @Composable
    fun LogItemCard(
        log: ScamLog,
        onFeedbackSubmit: (Int, String) -> Unit,
        onDelete: (Int) -> Unit
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            ),
            shape = RoundedCornerShape(12.dp),
            border = BorderStroke(1.dp, Color.White.copy(alpha = 0.05f))
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                // Header (Sender and Time)
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = "From: ${log.sender}",
                            fontWeight = FontWeight.Bold,
                            color = Color.White,
                            fontSize = 14.sp
                        )
                        Text(
                            text = formatTime(log.timestamp),
                            color = Color.Gray,
                            fontSize = 11.sp
                        )
                    }

                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        val scorePct = (log.confidence * 100).toInt()
                        Text(
                            text = "$scorePct% Match",
                            color = MaterialTheme.colorScheme.error,
                            fontWeight = FontWeight.Bold,
                            fontSize = 12.sp,
                            modifier = Modifier
                                .clip(RoundedCornerShape(6.dp))
                                .background(MaterialTheme.colorScheme.error.copy(alpha = 0.12f))
                                .padding(horizontal = 6.dp, vertical = 2.dp)
                        )

                        IconButton(
                            onClick = { onDelete(log.id) },
                            modifier = Modifier.size(24.dp)
                        ) {
                            Icon(
                                Icons.Default.Delete,
                                contentDescription = "Delete",
                                tint = Color.Gray,
                                modifier = Modifier.size(18.dp)
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

                // Text
                Text(
                    text = log.text,
                    fontSize = 13.sp,
                    color = Color.LightGray,
                    lineHeight = 18.sp
                )

                Spacer(modifier = Modifier.height(10.dp))

                // Active trigger tags
                if (log.triggeringTerms.isNotEmpty()) {
                    Text(
                        text = "Trigger Indicators:",
                        fontSize = 11.sp,
                        color = Color.Gray,
                        fontWeight = FontWeight.SemiBold
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        log.triggeringTerms.split(",").take(3).forEach { term ->
                            val clean = term.replace("'", "").trim()
                            if (clean.isNotEmpty()) {
                                Text(
                                    text = clean,
                                    fontSize = 10.sp,
                                    color = Color.White,
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(4.dp))
                                        .background(Color.White.copy(alpha = 0.05f))
                                        .border(1.dp, Color.White.copy(alpha = 0.1f), RoundedCornerShape(4.dp))
                                        .padding(horizontal = 6.dp, vertical = 2.dp)
                                )
                            }
                        }
                    }
                    Spacer(modifier = Modifier.height(12.dp))
                }

                Divider(color = Color.White.copy(alpha = 0.03f))

                Spacer(modifier = Modifier.height(8.dp))

                // Feedback Controls
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Feedback:",
                        fontSize = 12.sp,
                        color = Color.Gray
                    )

                    when (log.userFeedback) {
                        "pending" -> {
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                TextButton(
                                    onClick = { onFeedbackSubmit(log.id, "dismissed") },
                                    colors = ButtonDefaults.textButtonColors(
                                        contentColor = Color.LightGray
                                    ),
                                    modifier = Modifier.height(32.dp)
                                ) {
                                    Text("False Alarm", fontSize = 12.sp)
                                }
                                Button(
                                    onClick = { onFeedbackSubmit(log.id, "confirmed") },
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.2f),
                                        contentColor = MaterialTheme.colorScheme.primary
                                    ),
                                    modifier = Modifier.height(32.dp),
                                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 0.dp)
                                ) {
                                    Text("Confirm Scam", fontSize = 12.sp)
                                }
                            }
                        }
                        "confirmed" -> {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    Icons.Default.CheckCircle,
                                    contentDescription = "Confirmed",
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(16.dp)
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Scam Confirmed", fontSize = 12.sp, color = MaterialTheme.colorScheme.primary)
                            }
                        }
                        "dismissed" -> {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    Icons.Default.Info,
                                    contentDescription = "Dismissed",
                                    tint = Color.LightGray,
                                    modifier = Modifier.size(16.dp)
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("False Alarm Logged", fontSize = 12.sp, color = Color.LightGray)
                            }
                        }
                    }
                }
            }
        }
    }

    // Helper functions
    private fun hasRequiredPermissions(): Boolean {
        val receiveSms = ContextCompat.checkSelfPermission(this, Manifest.permission.RECEIVE_SMS)
        val readSms = ContextCompat.checkSelfPermission(this, Manifest.permission.READ_SMS)
        
        val notificationGranted = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED
        } else {
            true
        }

        return receiveSms == PackageManager.PERMISSION_GRANTED &&
                readSms == PackageManager.PERMISSION_GRANTED &&
                notificationGranted
    }

    private fun triggerPermissionRequest() {
        val permissions = mutableListOf(
            Manifest.permission.RECEIVE_SMS,
            Manifest.permission.READ_SMS
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        requestPermissionsLauncher.launch(permissions.toTypedArray())
    }

    private fun formatTime(timestamp: Long): String {
        val date = Date(timestamp)
        val format = SimpleDateFormat("dd MMM, hh:mm a", Locale.getDefault())
        return format.format(date)
    }
}
