import { 
  Briefcase, 
  Users, 
  GitBranch, 
  LayoutDashboard,
  Settings,
  HelpCircle,
  Sparkles,
  ChevronLeft,
  ChevronRight
} from 'lucide-react'

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'jobs', label: 'Jobs', icon: Briefcase },
  { id: 'candidates', label: 'Candidates', icon: Users },
  { id: 'pipelines', label: 'Active Pipelines', icon: GitBranch },
]

const BOTTOM_NAV = [
  { id: 'settings', label: 'Settings', icon: Settings },
  { id: 'help', label: 'Help & Support', icon: HelpCircle },
]

function Sidebar({ activeView, onViewChange, isCollapsed, onToggleCollapse, pipelineCount = 0 }) {
  return (
    <aside 
      className={`
        fixed left-0 top-0 h-screen z-40 
        flex flex-col
        bg-surface-900/80 backdrop-blur-xl
        border-r border-surface-800
        transition-all duration-300 ease-in-out
        ${isCollapsed ? 'w-20' : 'w-64'}
      `}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 h-16 border-b border-surface-800">
        <div className="relative">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-600 to-indigo-400 flex items-center justify-center shadow-lg shadow-indigo-500/30">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-emerald-500 rounded-full border-2 border-surface-900 flex items-center justify-center">
            <span className="text-[8px] font-bold text-white">AI</span>
          </div>
        </div>
        {!isCollapsed && (
          <div className="overflow-hidden">
            <h1 className="text-lg font-bold text-white truncate">RecruitAI</h1>
            <p className="text-xs text-surface-500 truncate">Intelligent Hiring</p>
          </div>
        )}
      </div>

      {/* Main Navigation */}
      <nav className="flex-1 py-6 px-3 space-y-1 overflow-y-auto hide-scrollbar">
        <div className={`text-xs font-semibold text-surface-500 uppercase tracking-wider mb-3 ${isCollapsed ? 'text-center' : 'px-3'}`}>
          {isCollapsed ? '•••' : 'Navigation'}
        </div>
        
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon
          const isActive = activeView === item.id
          const showBadge = item.id === 'pipelines' && pipelineCount > 0

          return (
            <button
              key={item.id}
              onClick={() => onViewChange(item.id)}
              className={`
                w-full flex items-center gap-3 px-3 py-3 rounded-xl
                transition-all duration-200 group relative
                ${isActive 
                  ? 'bg-gradient-to-r from-indigo-600/20 to-transparent text-white border-l-2 border-indigo-500' 
                  : 'text-surface-400 hover:text-white hover:bg-white/5'
                }
                ${isCollapsed ? 'justify-center' : ''}
              `}
              title={isCollapsed ? item.label : undefined}
            >
              <Icon className={`w-5 h-5 flex-shrink-0 transition-transform duration-200 ${isActive ? 'text-indigo-400' : 'group-hover:scale-110'}`} />
              
              {!isCollapsed && (
                <span className="font-medium truncate">{item.label}</span>
              )}

              {showBadge && (
                <span className={`
                  flex items-center justify-center
                  min-w-[20px] h-5 px-1.5
                  bg-indigo-500 text-white text-xs font-bold rounded-full
                  ${isCollapsed ? 'absolute -top-1 -right-1' : 'ml-auto'}
                `}>
                  {pipelineCount}
                </span>
              )}

              {/* Tooltip for collapsed state */}
              {isCollapsed && (
                <div className="
                  absolute left-full ml-3 px-3 py-2
                  bg-surface-800 text-white text-sm font-medium
                  rounded-lg shadow-xl
                  opacity-0 invisible group-hover:opacity-100 group-hover:visible
                  transition-all duration-200
                  whitespace-nowrap z-50
                  border border-surface-700
                ">
                  {item.label}
                  <div className="absolute left-0 top-1/2 -translate-x-1 -translate-y-1/2 w-2 h-2 bg-surface-800 rotate-45 border-l border-b border-surface-700" />
                </div>
              )}
            </button>
          )
        })}
      </nav>

      {/* Bottom Navigation */}
      <div className="py-4 px-3 border-t border-surface-800 space-y-1">
        {BOTTOM_NAV.map((item) => {
          const Icon = item.icon
          
          return (
            <button
              key={item.id}
              onClick={() => onViewChange(item.id)}
              className={`
                w-full flex items-center gap-3 px-3 py-2.5 rounded-xl
                text-surface-500 hover:text-surface-300 hover:bg-white/5
                transition-all duration-200 group
                ${isCollapsed ? 'justify-center' : ''}
              `}
              title={isCollapsed ? item.label : undefined}
            >
              <Icon className="w-5 h-5 flex-shrink-0" />
              {!isCollapsed && (
                <span className="text-sm truncate">{item.label}</span>
              )}
            </button>
          )
        })}
      </div>

      {/* Collapse Toggle */}
      <div className="p-3 border-t border-surface-800">
        <button
          onClick={onToggleCollapse}
          className="
            w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-xl
            text-surface-500 hover:text-white hover:bg-white/5
            transition-all duration-200
          "
        >
          {isCollapsed ? (
            <ChevronRight className="w-5 h-5" />
          ) : (
            <>
              <ChevronLeft className="w-5 h-5" />
              <span className="text-sm">Collapse</span>
            </>
          )}
        </button>
      </div>
    </aside>
  )
}

export default Sidebar
