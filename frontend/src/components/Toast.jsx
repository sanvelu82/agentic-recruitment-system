import { useState, useEffect, createContext, useContext, useCallback } from 'react'
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react'

// Toast Context
const ToastContext = createContext(null)

// Toast Types Configuration
const TOAST_TYPES = {
  success: {
    icon: CheckCircle,
    className: 'border-emerald-500/50 bg-emerald-500/10',
    iconClass: 'text-emerald-400',
    progressClass: 'bg-emerald-500',
  },
  error: {
    icon: AlertCircle,
    className: 'border-red-500/50 bg-red-500/10',
    iconClass: 'text-red-400',
    progressClass: 'bg-red-500',
  },
  warning: {
    icon: AlertTriangle,
    className: 'border-amber-500/50 bg-amber-500/10',
    iconClass: 'text-amber-400',
    progressClass: 'bg-amber-500',
  },
  info: {
    icon: Info,
    className: 'border-indigo-500/50 bg-indigo-500/10',
    iconClass: 'text-indigo-400',
    progressClass: 'bg-indigo-500',
  },
}

// Individual Toast Component
function ToastItem({ toast, onDismiss }) {
  const [isExiting, setIsExiting] = useState(false)
  const [progress, setProgress] = useState(100)
  
  const { type = 'info', message, title, duration = 5000, id } = toast
  const config = TOAST_TYPES[type] || TOAST_TYPES.info
  const Icon = config.icon

  useEffect(() => {
    if (duration <= 0) return

    const startTime = Date.now()
    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime
      const remaining = Math.max(0, 100 - (elapsed / duration) * 100)
      setProgress(remaining)
      
      if (remaining <= 0) {
        clearInterval(interval)
        handleDismiss()
      }
    }, 50)

    return () => clearInterval(interval)
  }, [duration])

  const handleDismiss = () => {
    setIsExiting(true)
    setTimeout(() => onDismiss(id), 300)
  }

  return (
    <div
      className={`
        relative overflow-hidden w-96 rounded-xl border backdrop-blur-xl
        shadow-2xl shadow-black/20
        ${config.className}
        ${isExiting ? 'animate-slide-out' : 'animate-slide-in'}
      `}
    >
      <div className="flex items-start gap-3 p-4">
        <div className={`flex-shrink-0 mt-0.5 ${config.iconClass}`}>
          <Icon className="w-5 h-5" />
        </div>
        <div className="flex-1 min-w-0">
          {title && (
            <p className="text-sm font-semibold text-white mb-0.5">{title}</p>
          )}
          <p className="text-sm text-surface-300">{message}</p>
        </div>
        <button
          onClick={handleDismiss}
          className="flex-shrink-0 p-1 rounded-lg text-surface-400 hover:text-white hover:bg-white/10 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
      
      {/* Progress Bar */}
      {duration > 0 && (
        <div className="h-1 bg-surface-800">
          <div
            className={`h-full transition-all duration-100 ease-linear ${config.progressClass}`}
            style={{ width: `${progress}%` }}
          />
        </div>
      )}
    </div>
  )
}

// Toast Container
function ToastContainer({ toasts, onDismiss }) {
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-3">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />
      ))}
    </div>
  )
}

// Toast Provider
export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const addToast = useCallback((toast) => {
    const id = Date.now() + Math.random()
    setToasts((prev) => [...prev, { ...toast, id }])
    return id
  }, [])

  const dismissToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  const toast = useCallback({
    success: (message, options = {}) => 
      addToast({ type: 'success', message, ...options }),
    error: (message, options = {}) => 
      addToast({ type: 'error', message, ...options }),
    warning: (message, options = {}) => 
      addToast({ type: 'warning', message, ...options }),
    info: (message, options = {}) => 
      addToast({ type: 'info', message, ...options }),
  }, [addToast])

  return (
    <ToastContext.Provider value={{ toast, dismissToast }}>
      {children}
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </ToastContext.Provider>
  )
}

// Hook to use toast
export function useToast() {
  const context = useContext(ToastContext)
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider')
  }
  return context.toast
}

export default ToastProvider
