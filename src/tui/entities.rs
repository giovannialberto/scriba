use crate::core::rebuild_world_from_entities;
use crate::entities::EntityRegistry;
use anyhow::Result;
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Clear, Paragraph, Row, Table, Wrap},
    Frame,
};

use super::chat::ACCENT;
use super::app::{Dashboard, DashboardAction, DashboardView};

// ─────────────────────────────────────────────────────────────────────────────
// Entity types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq)]
pub(super) enum EntityMode {
    Browse,
    Editing,
    Adding,
    DeleteConfirm,
    MergeSelectTarget,
    MergeConfirm,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub(super) enum EntityEditField {
    Name,
    Type,
    Context,
}

pub(super) const ENTITY_TYPES: &[&str] = &["person", "organization", "project", "other"];

// ─────────────────────────────────────────────────────────────────────────────
// Entity methods on Dashboard
// ─────────────────────────────────────────────────────────────────────────────

impl Dashboard {
    pub(super) fn load_entities(&mut self) -> Result<()> {
        self.entities = self.db.list_entities(None, None)?;
        if !self.entities.is_empty() && self.entity_table_state.selected().is_none() {
            self.entity_table_state.select(Some(0));
        }
        Ok(())
    }

    pub(super) async fn handle_entities_keys(&mut self, key_code: KeyCode) -> Result<DashboardAction> {
        match self.entity_mode {
            EntityMode::Browse => {
                if self.show_entity_detail {
                    match key_code {
                        KeyCode::Esc => {
                            self.show_entity_detail = false;
                            self.selected_entity = None;
                        }
                        _ => {}
                    }
                    return Ok(DashboardAction::Continue);
                }

                match key_code {
                    KeyCode::Esc | KeyCode::Char('q') => {
                        self.current_view = DashboardView::Main;
                        self.show_entity_detail = false;
                        self.selected_entity = None;
                    }
                    KeyCode::Up => {
                        self.entity_navigate_up();
                    }
                    KeyCode::Down => {
                        self.entity_navigate_down();
                    }
                    KeyCode::Enter => {
                        if let Some(idx) = self.entity_table_state.selected() {
                            if let Some(entity) = self.entities.get(idx) {
                                self.selected_entity = Some(entity.clone());
                                self.show_entity_detail = true;
                            }
                        }
                    }
                    KeyCode::Char('e') | KeyCode::Char('E') => {
                        self.start_entity_edit();
                    }
                    KeyCode::Char('d') | KeyCode::Char('D') | KeyCode::Delete => {
                        if let Some(idx) = self.entity_table_state.selected() {
                            if let Some(entity) = self.entities.get(idx) {
                                self.selected_entity = Some(entity.clone());
                                self.confirm_selection = 1; // default to No
                                self.entity_mode = EntityMode::DeleteConfirm;
                            }
                        }
                    }
                    KeyCode::Char('m') | KeyCode::Char('M') => {
                        if let Some(idx) = self.entity_table_state.selected() {
                            if let Some(entity) = self.entities.get(idx) {
                                self.merge_source_entity = Some(entity.clone());
                                self.entity_mode = EntityMode::MergeSelectTarget;
                            }
                        }
                    }
                    KeyCode::Char('a') | KeyCode::Char('A') => {
                        self.start_entity_add();
                    }
                    KeyCode::Char('r') | KeyCode::Char('R') => {
                        self.load_entities()?;
                    }
                    _ => {}
                }
            }
            EntityMode::Adding => {
                match key_code {
                    KeyCode::Esc => {
                        self.entity_mode = EntityMode::Browse;
                    }
                    KeyCode::Tab | KeyCode::Down => {
                        self.entity_add_field = match self.entity_add_field {
                            EntityEditField::Name => EntityEditField::Type,
                            EntityEditField::Type => EntityEditField::Context,
                            EntityEditField::Context => EntityEditField::Name,
                        };
                    }
                    KeyCode::BackTab | KeyCode::Up => {
                        self.entity_add_field = match self.entity_add_field {
                            EntityEditField::Name => EntityEditField::Context,
                            EntityEditField::Type => EntityEditField::Name,
                            EntityEditField::Context => EntityEditField::Type,
                        };
                    }
                    KeyCode::Enter => {
                        if self.entity_add_field == EntityEditField::Type {
                            // Cycle type on Enter
                            let current_idx = ENTITY_TYPES.iter().position(|t| *t == self.entity_add_type).unwrap_or(0);
                            self.entity_add_type = ENTITY_TYPES[(current_idx + 1) % ENTITY_TYPES.len()].to_string();
                        } else if !self.entity_add_name.trim().is_empty() {
                            // Save if name is not empty
                            self.save_entity_add()?;
                            self.entity_mode = EntityMode::Browse;
                        }
                    }
                    KeyCode::Char(' ') if self.entity_add_field == EntityEditField::Type => {
                        let current_idx = ENTITY_TYPES.iter().position(|t| *t == self.entity_add_type).unwrap_or(0);
                        self.entity_add_type = ENTITY_TYPES[(current_idx + 1) % ENTITY_TYPES.len()].to_string();
                    }
                    KeyCode::Backspace => {
                        match self.entity_add_field {
                            EntityEditField::Name => { self.entity_add_name.pop(); }
                            EntityEditField::Type => {}
                            EntityEditField::Context => { self.entity_add_context.pop(); }
                        }
                    }
                    KeyCode::Char(c) => {
                        match self.entity_add_field {
                            EntityEditField::Name => self.entity_add_name.push(c),
                            EntityEditField::Type => {}
                            EntityEditField::Context => self.entity_add_context.push(c),
                        }
                    }
                    _ => {}
                }
            }
            EntityMode::Editing => {
                match key_code {
                    KeyCode::Esc => {
                        // Cancel edit — discard changes
                        self.selected_entity = None;
                        self.entity_mode = EntityMode::Browse;
                    }
                    KeyCode::Tab | KeyCode::Down => {
                        self.entity_edit_field = match self.entity_edit_field {
                            EntityEditField::Name => EntityEditField::Type,
                            EntityEditField::Type => EntityEditField::Context,
                            EntityEditField::Context => EntityEditField::Name,
                        };
                    }
                    KeyCode::BackTab | KeyCode::Up => {
                        self.entity_edit_field = match self.entity_edit_field {
                            EntityEditField::Name => EntityEditField::Context,
                            EntityEditField::Type => EntityEditField::Name,
                            EntityEditField::Context => EntityEditField::Type,
                        };
                    }
                    KeyCode::Enter => {
                        if self.entity_edit_field == EntityEditField::Type {
                            self.cycle_entity_type();
                        } else if self.entity_edit_field == EntityEditField::Context {
                            // Save and exit from the last field
                            self.save_entity_edit()?;
                            self.entity_mode = EntityMode::Browse;
                        } else {
                            // Move to next field
                            self.entity_edit_field = match self.entity_edit_field {
                                EntityEditField::Name => EntityEditField::Type,
                                EntityEditField::Type => EntityEditField::Context,
                                EntityEditField::Context => EntityEditField::Name,
                            };
                        }
                    }
                    KeyCode::Char(' ') if self.entity_edit_field == EntityEditField::Type => {
                        self.cycle_entity_type();
                    }
                    KeyCode::Char(c) => {
                        match self.entity_edit_field {
                            EntityEditField::Name => self.entity_edit_name.push(c),
                            EntityEditField::Context => self.entity_edit_context.push(c),
                            EntityEditField::Type => {} // Type is cycled, not typed
                        }
                    }
                    KeyCode::Backspace => {
                        match self.entity_edit_field {
                            EntityEditField::Name => { self.entity_edit_name.pop(); }
                            EntityEditField::Context => { self.entity_edit_context.pop(); }
                            EntityEditField::Type => {}
                        }
                    }
                    _ => {}
                }
            }
            EntityMode::DeleteConfirm => {
                match key_code {
                    KeyCode::Up | KeyCode::Char('k') => {
                        self.confirm_selection = 0;
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        self.confirm_selection = 1;
                    }
                    KeyCode::Enter => {
                        if self.confirm_selection == 0 {
                            self.perform_entity_delete()?;
                        } else {
                            self.selected_entity = None;
                        }
                        self.entity_mode = EntityMode::Browse;
                    }
                    KeyCode::Esc => {
                        self.selected_entity = None;
                        self.entity_mode = EntityMode::Browse;
                    }
                    _ => {}
                }
            }
            EntityMode::MergeSelectTarget => {
                match key_code {
                    KeyCode::Up => {
                        self.entity_navigate_up();
                    }
                    KeyCode::Down => {
                        self.entity_navigate_down();
                    }
                    KeyCode::Enter => {
                        if let Some(idx) = self.entity_table_state.selected() {
                            if let Some(target) = self.entities.get(idx) {
                                let source_id = self.merge_source_entity.as_ref()
                                    .and_then(|e| e.id);
                                // Prevent merge when source has no DB id or target is the same entity
                                if let Some(src_id) = source_id {
                                    if Some(src_id) != target.id {
                                        self.selected_entity = Some(target.clone());
                                        self.confirm_selection = 1; // default to No
                                        self.entity_mode = EntityMode::MergeConfirm;
                                    }
                                }
                            }
                        }
                    }
                    KeyCode::Esc => {
                        self.merge_source_entity = None;
                        self.entity_mode = EntityMode::Browse;
                    }
                    _ => {}
                }
            }
            EntityMode::MergeConfirm => {
                match key_code {
                    KeyCode::Up | KeyCode::Char('k') => {
                        self.confirm_selection = 0;
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        self.confirm_selection = 1;
                    }
                    KeyCode::Enter => {
                        if self.confirm_selection == 0 {
                            self.perform_entity_merge()?;
                        } else {
                            self.selected_entity = None;
                            self.merge_source_entity = None;
                        }
                        self.entity_mode = EntityMode::Browse;
                    }
                    KeyCode::Esc => {
                        self.selected_entity = None;
                        self.merge_source_entity = None;
                        self.entity_mode = EntityMode::Browse;
                    }
                    _ => {}
                }
            }
        }
        Ok(DashboardAction::Continue)
    }

    pub(super) fn entity_navigate_up(&mut self) {
        let i = match self.entity_table_state.selected() {
            Some(i) => if i == 0 { self.entities.len().saturating_sub(1) } else { i - 1 },
            None => 0,
        };
        self.entity_table_state.select(Some(i));
    }

    pub(super) fn entity_navigate_down(&mut self) {
        let i = match self.entity_table_state.selected() {
            Some(i) => if i >= self.entities.len().saturating_sub(1) { 0 } else { i + 1 },
            None => 0,
        };
        self.entity_table_state.select(Some(i));
    }

    pub(super) fn start_entity_edit(&mut self) {
        if let Some(idx) = self.entity_table_state.selected() {
            if let Some(entity) = self.entities.get(idx) {
                self.selected_entity = Some(entity.clone());
                self.entity_edit_name = entity.canonical_name.clone();
                self.entity_edit_type = entity.entity_type.clone();
                self.entity_edit_context = entity.context.clone().unwrap_or_default();
                self.entity_edit_field = EntityEditField::Name;
                self.entity_mode = EntityMode::Editing;
            }
        }
    }

    pub(super) fn cycle_entity_type(&mut self) {
        let current_idx = ENTITY_TYPES.iter()
            .position(|t| *t == self.entity_edit_type)
            .unwrap_or(0);
        let next_idx = (current_idx + 1) % ENTITY_TYPES.len();
        self.entity_edit_type = ENTITY_TYPES[next_idx].to_string();
    }

    pub(super) fn save_entity_edit(&mut self) -> Result<()> {
        if let Some(entity) = &self.selected_entity {
            let entity_id = match entity.id {
                Some(id) => id,
                None => return Ok(()),
            };

            let mut registry = EntityRegistry::new(&mut self.db);

            // Update name if changed
            if self.entity_edit_name != entity.canonical_name && !self.entity_edit_name.is_empty() {
                registry.rename_entity(entity_id, &self.entity_edit_name)?;
            }

            // Update type if changed
            if self.entity_edit_type != entity.entity_type {
                registry.update_entity_type(entity_id, &self.entity_edit_type)?;
            }

            // Update context if changed
            let old_context = entity.context.as_deref().unwrap_or("");
            if self.entity_edit_context != old_context {
                let ctx = if self.entity_edit_context.is_empty() { "" } else { &self.entity_edit_context };
                registry.update_entity_context(entity_id, ctx)?;
            }

            drop(registry);
            let _ = rebuild_world_from_entities(&self.db);
            self.load_entities()?;
            self.selected_entity = None;
        }
        Ok(())
    }

    pub(super) fn perform_entity_delete(&mut self) -> Result<()> {
        if let Some(entity) = &self.selected_entity {
            if let Some(id) = entity.id {
                self.db.delete_entity(id)?;
                let _ = rebuild_world_from_entities(&self.db);
                self.load_entities()?;
                // Adjust selection if needed
                if let Some(selected) = self.entity_table_state.selected() {
                    if selected >= self.entities.len() && !self.entities.is_empty() {
                        self.entity_table_state.select(Some(self.entities.len() - 1));
                    }
                }
            }
        }
        self.selected_entity = None;
        Ok(())
    }

    pub(super) fn perform_entity_merge(&mut self) -> Result<()> {
        let source_id = self.merge_source_entity.as_ref().and_then(|e| e.id);
        let target_id = self.selected_entity.as_ref().and_then(|e| e.id);

        if let (Some(src), Some(tgt)) = (source_id, target_id) {
            let mut registry = EntityRegistry::new(&mut self.db);
            registry.merge_entities(src, tgt)?;
            drop(registry);
            let _ = rebuild_world_from_entities(&self.db);
            self.load_entities()?;
        }

        self.merge_source_entity = None;
        self.selected_entity = None;
        Ok(())
    }

    pub(super) fn start_entity_add(&mut self) {
        self.entity_add_name.clear();
        self.entity_add_type = "person".to_string();
        self.entity_add_context.clear();
        self.entity_add_aliases.clear();
        self.entity_add_field = EntityEditField::Name;
        self.entity_mode = EntityMode::Adding;
    }

    pub(super) fn save_entity_add(&mut self) -> Result<()> {
        if self.entity_add_name.trim().is_empty() {
            return Ok(());
        }
        let mut registry = EntityRegistry::new(&mut self.db);
        let entity = registry.create_entity(
            self.entity_add_type.trim(),
            self.entity_add_name.trim(),
            if self.entity_add_context.trim().is_empty() {
                None
            } else {
                Some(self.entity_add_context.trim())
            },
        )?;
        // Add aliases if provided
        if let Some(id) = entity.id {
            for alias in self.entity_add_aliases.split(',') {
                let alias = alias.trim();
                if !alias.is_empty() {
                    registry.add_entity_alias(id, alias)?;
                }
            }
        }
        drop(registry);
        let _ = rebuild_world_from_entities(&self.db);
        self.load_entities()?;
        // Select the new entity (should be last or find by name)
        if let Some(pos) = self.entities.iter().position(|e| e.canonical_name == self.entity_add_name.trim()) {
            self.entity_table_state.select(Some(pos));
        }
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Entity rendering
    // ─────────────────────────────────────────────────────────────────────────

    pub(super) fn render_entities_view(&mut self, f: &mut Frame, area: ratatui::layout::Rect) {
        use ratatui::text::{Line, Span};

        // Full-screen borderless layout: header + table + footer
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),  // Header
                Constraint::Min(8),    // Entity table
                Constraint::Length(2), // Footer
            ])
            .split(area);

        // ── Header ──────────────────────────────────────────────────
        let people_count = self.entities.iter().filter(|e| e.entity_type == "person").count();
        let org_count = self.entities.iter().filter(|e| e.entity_type == "organization").count();
        let project_count = self.entities.iter().filter(|e| e.entity_type == "project").count();

        let mut count_parts: Vec<String> = Vec::new();
        if people_count > 0 { count_parts.push(format!("{} people", people_count)); }
        if org_count > 0 { count_parts.push(format!("{} orgs", org_count)); }
        if project_count > 0 { count_parts.push(format!("{} projects", project_count)); }
        let counts_suffix = if count_parts.is_empty() {
            String::new()
        } else {
            format!(" \u{00B7} {}", count_parts.join(" \u{00B7} "))
        };

        let header_line = Line::from(vec![
            Span::raw("  "),
            Span::styled("Scriba's World", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(counts_suffix, Style::default().fg(Color::DarkGray)),
        ]);
        f.render_widget(Paragraph::new(header_line), Rect { x: main_chunks[0].x, y: main_chunks[0].y, width: main_chunks[0].width, height: 1 });

        let sep = "\u{2500}".repeat(main_chunks[0].width as usize);
        f.render_widget(
            Paragraph::new(sep).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: main_chunks[0].x, y: main_chunks[0].y + 1, width: main_chunks[0].width, height: 1 },
        );

        // ── Entity table ────────────────────────────────────────────
        let header_cells = ["ID", "Type", "Name", "Aliases", "Context", "Mentions"]
            .iter()
            .map(|h| {
                Cell::from(*h).style(
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )
            });

        let header_row = Row::new(header_cells)
            .style(Style::default())
            .height(1)
            .bottom_margin(1);

        let merge_source_id = self.merge_source_entity.as_ref().and_then(|e| e.id);
        let rows: Vec<Row> = self
            .entities
            .iter()
            .map(|entity| {
                let is_merge_source = self.entity_mode == EntityMode::MergeSelectTarget
                    && merge_source_id == entity.id;

                let aliases = entity.aliases_list().join(", ");
                let aliases_display = if aliases.is_empty() {
                    "-".to_string()
                } else if aliases.chars().count() > 20 {
                    let truncated: String = aliases.chars().take(17).collect();
                    format!("{}...", truncated)
                } else {
                    aliases
                };

                let context_display = entity
                    .context
                    .as_ref()
                    .map(|c| {
                        if c.chars().count() > 30 {
                            let truncated: String = c.chars().take(27).collect();
                            format!("{}...", truncated)
                        } else {
                            c.clone()
                        }
                    })
                    .unwrap_or_else(|| "-".to_string());

                let type_color = if is_merge_source {
                    Color::DarkGray
                } else {
                    match entity.entity_type.as_str() {
                        "person" => Color::Green,
                        "organization" => Color::Blue,
                        _ => Color::Gray,
                    }
                };

                let name_style = if is_merge_source {
                    Style::default().fg(Color::DarkGray)
                } else {
                    Style::default().add_modifier(Modifier::BOLD)
                };

                let dim = if is_merge_source { Style::default().fg(Color::DarkGray) } else { Style::default().fg(Color::Gray) };

                let cells = vec![
                    Cell::from(entity.id.unwrap_or(0).to_string()),
                    Cell::from(entity.entity_type.clone()).style(Style::default().fg(type_color)),
                    Cell::from(entity.canonical_name.clone()).style(name_style),
                    Cell::from(aliases_display).style(dim),
                    Cell::from(context_display).style(dim),
                    Cell::from(entity.mention_count.to_string())
                        .style(if is_merge_source { Style::default().fg(Color::DarkGray) } else { Style::default().fg(Color::Yellow) }),
                ];

                Row::new(cells).height(1).bottom_margin(0)
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Length(4),   // ID
                Constraint::Length(12),  // Type
                Constraint::Length(20),  // Name
                Constraint::Length(22),  // Aliases
                Constraint::Min(20),     // Context
                Constraint::Length(8),   // Mentions
            ],
        )
        .header(header_row)
        .block(Block::default())
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("\u{25B6} ");

        f.render_stateful_widget(table, main_chunks[1], &mut self.entity_table_state);

        // ── Footer ──────────────────────────────────────────────────
        let sep2 = "\u{2500}".repeat(main_chunks[2].width as usize);
        f.render_widget(
            Paragraph::new(sep2).style(Style::default().fg(Color::Indexed(237))),
            Rect { x: main_chunks[2].x, y: main_chunks[2].y, width: main_chunks[2].width, height: 1 },
        );

        let footer_area = Rect { x: main_chunks[2].x, y: main_chunks[2].y + 1, width: main_chunks[2].width, height: 1 };
        let version = env!("CARGO_PKG_VERSION");
        let entity_count = self.entities.len();
        let left_spans = vec![
            Span::styled(" \u{25B8} ", Style::default().fg(ACCENT)),
            Span::styled(
                format!("scriba \u{00B7} v{} \u{00B7} {} entities", version, entity_count),
                Style::default().fg(Color::DarkGray),
            ),
        ];

        let right_spans: Vec<Span> = match self.entity_mode {
            EntityMode::Browse => vec![
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("A", Style::default().fg(Color::White)),
                Span::styled("] Add  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("E", Style::default().fg(Color::White)),
                Span::styled("] Edit  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("D", Style::default().fg(Color::White)),
                Span::styled("] Del  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("M", Style::default().fg(Color::White)),
                Span::styled("] Merge  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("R", Style::default().fg(Color::White)),
                Span::styled("] Refresh  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Esc", Style::default().fg(Color::White)),
                Span::styled("] Back ", Style::default().fg(Color::DarkGray)),
            ],
            EntityMode::MergeSelectTarget => vec![
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Enter", Style::default().fg(Color::White)),
                Span::styled("] Confirm  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Esc", Style::default().fg(Color::White)),
                Span::styled("] Cancel ", Style::default().fg(Color::DarkGray)),
            ],
            EntityMode::DeleteConfirm => vec![
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Y", Style::default().fg(Color::White)),
                Span::styled("] Confirm  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("N", Style::default().fg(Color::White)),
                Span::styled("] Cancel ", Style::default().fg(Color::DarkGray)),
            ],
            EntityMode::MergeConfirm => vec![
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Y", Style::default().fg(Color::White)),
                Span::styled("] Confirm  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("N", Style::default().fg(Color::White)),
                Span::styled("] Cancel ", Style::default().fg(Color::DarkGray)),
            ],
            EntityMode::Adding | EntityMode::Editing => vec![
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Tab", Style::default().fg(Color::White)),
                Span::styled("] Next field  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Space", Style::default().fg(Color::White)),
                Span::styled("] Cycle type  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(Color::DarkGray)),
                Span::styled("Esc", Style::default().fg(Color::White)),
                Span::styled("] Done ", Style::default().fg(Color::DarkGray)),
            ],
        };

        let left_w: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();
        let right_w: usize = right_spans.iter().map(|s| s.content.chars().count()).sum();
        let gap = (footer_area.width as usize).saturating_sub(left_w + right_w);
        let mut spans = left_spans;
        spans.push(Span::raw(" ".repeat(gap)));
        spans.extend(right_spans);
        f.render_widget(Paragraph::new(Line::from(spans)), footer_area);

        // Popups (overlays stay unchanged)
        if self.show_entity_detail {
            self.render_entity_detail_popup(f, area);
        }
        if self.entity_mode == EntityMode::Adding {
            self.render_entity_add_popup(f, area);
        }
        if self.entity_mode == EntityMode::Editing {
            self.render_entity_edit_popup(f, area);
        }
        if self.entity_mode == EntityMode::DeleteConfirm {
            self.render_entity_delete_confirm(f, area);
        }
        if self.entity_mode == EntityMode::MergeConfirm {
            self.render_entity_merge_confirm(f, area);
        }
    }

    pub(super) fn render_entity_detail_popup(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let popup_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(10),
                Constraint::Percentage(80),
                Constraint::Percentage(10),
            ])
            .split(area)[1];

        let popup_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(10),
                Constraint::Percentage(80),
                Constraint::Percentage(10),
            ])
            .split(popup_area)[1];

        f.render_widget(Clear, popup_area);

        if let Some(entity) = &self.selected_entity {
            let aliases = entity.aliases_list().join(", ");
            let aliases_display = if aliases.is_empty() {
                "-".to_string()
            } else {
                aliases
            };

            let context_display = entity
                .context
                .as_ref()
                .cloned()
                .unwrap_or_else(|| "(no context)".to_string());

            let type_label = match entity.entity_type.as_str() {
                "person" => "person",
                "organization" => "org",
                _ => "other",
            };

            let content = vec![
                Line::from(vec![
                    Span::styled(
                        format!("[{}] ", type_label),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(
                        &entity.canonical_name,
                        Style::default()
                            .fg(ACCENT)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Type: ", Style::default().fg(Color::Green)),
                    Span::styled(
                        &entity.entity_type,
                        Style::default().fg(Color::White),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("ID: ", Style::default().fg(Color::Green)),
                    Span::styled(
                        entity.id.unwrap_or(0).to_string(),
                        Style::default().fg(Color::White),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Mentions: ", Style::default().fg(Color::Green)),
                    Span::styled(
                        entity.mention_count.to_string(),
                        Style::default().fg(Color::Yellow),
                    ),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Aliases: ", Style::default().fg(Color::Green)),
                    Span::styled(
                        aliases_display,
                        Style::default().fg(Color::Gray),
                    ),
                ]),
                Line::from(""),
                Line::from(vec![Span::styled(
                    "Context:",
                    Style::default().fg(Color::Green),
                )]),
                Line::from(vec![Span::styled(
                    context_display,
                    Style::default().fg(Color::White),
                )]),
                Line::from(""),
                Line::from(""),
                Line::from(vec![Span::styled(
                    "Press ESC to close | E to edit | D to delete | M to merge",
                    Style::default().fg(Color::Blue),
                )]),
            ];

            let detail_paragraph = Paragraph::new(content)
                .style(Style::default().fg(Color::White))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(format!("Entity Details - {}", entity.canonical_name))
                        .title_style(
                            Style::default()
                                .fg(ACCENT)
                                .add_modifier(Modifier::BOLD),
                        )
                        .border_style(Style::default().fg(ACCENT)),
                )
                .wrap(Wrap { trim: true });

            f.render_widget(detail_paragraph, popup_area);
        }
    }

    pub(super) fn render_entity_add_popup(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let popup_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(15),
                Constraint::Percentage(70),
                Constraint::Percentage(15),
            ])
            .split(area)[1];
        let popup_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(15),
                Constraint::Percentage(70),
                Constraint::Percentage(15),
            ])
            .split(popup_area)[1];

        f.render_widget(Clear, popup_area);

        let cursor = "\u{2588}";
        let name_style = if self.entity_add_field == EntityEditField::Name {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };
        let context_style = if self.entity_add_field == EntityEditField::Context {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        let name_display = if self.entity_add_field == EntityEditField::Name {
            format!("{}{}", self.entity_add_name, cursor)
        } else if self.entity_add_name.is_empty() {
            "(type a name)".to_string()
        } else {
            self.entity_add_name.clone()
        };

        let type_display: Vec<Span> = ENTITY_TYPES.iter().map(|t| {
            if *t == self.entity_add_type {
                Span::styled(format!(" [{}] ", t), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))
            } else {
                Span::styled(format!("  {}  ", t), Style::default().fg(Color::Gray))
            }
        }).collect();

        let context_display = if self.entity_add_field == EntityEditField::Context {
            format!("{}{}", self.entity_add_context, cursor)
        } else if self.entity_add_context.is_empty() {
            "(optional)".to_string()
        } else {
            self.entity_add_context.clone()
        };

        let content = vec![
            Line::from(vec![
                Span::styled("Name: ", Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![Span::styled(name_display, name_style)]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Type: ", Style::default().fg(Color::Green)),
            ]),
            Line::from(type_display),
            Line::from(""),
            Line::from(vec![
                Span::styled("Context: ", Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![Span::styled(context_display, context_style)]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Tab/\u{2191}\u{2193}: Switch Field | Space: Cycle Type | Enter: Save | Esc: Cancel", Style::default().fg(Color::DarkGray)),
            ]),
        ];

        let add_paragraph = Paragraph::new(content)
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" ADD NEW ENTITY ")
                    .title_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
                    .border_style(Style::default().fg(Color::Yellow)),
            )
            .wrap(Wrap { trim: true });

        f.render_widget(add_paragraph, popup_area);
    }

    pub(super) fn render_entity_edit_popup(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let popup_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(15),
                Constraint::Percentage(70),
                Constraint::Percentage(15),
            ])
            .split(area)[1];
        let popup_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(15),
                Constraint::Percentage(70),
                Constraint::Percentage(15),
            ])
            .split(popup_area)[1];

        f.render_widget(Clear, popup_area);

        let name_style = if self.entity_edit_field == EntityEditField::Name {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };
        let context_style = if self.entity_edit_field == EntityEditField::Context {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        let cursor = "\u{2588}";
        let name_display = if self.entity_edit_field == EntityEditField::Name {
            format!("{}{}", self.entity_edit_name, cursor)
        } else {
            self.entity_edit_name.clone()
        };

        let type_display: Vec<Span> = ENTITY_TYPES.iter().map(|t| {
            if *t == self.entity_edit_type {
                Span::styled(format!(" [{}] ", t), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))
            } else {
                Span::styled(format!("  {}  ", t), Style::default().fg(Color::Gray))
            }
        }).collect();

        let context_display = if self.entity_edit_field == EntityEditField::Context {
            format!("{}{}", self.entity_edit_context, cursor)
        } else {
            if self.entity_edit_context.is_empty() { "(empty)".to_string() } else { self.entity_edit_context.clone() }
        };

        let content = vec![
            Line::from(vec![
                Span::styled("Name: ", Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![Span::styled(name_display, name_style)]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Type: ", Style::default().fg(Color::Green)),
            ]),
            Line::from(type_display),
            Line::from(""),
            Line::from(vec![
                Span::styled("Context: ", Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![Span::styled(context_display, context_style)]),
        ];

        let edit_paragraph = Paragraph::new(content)
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Edit Entity")
                    .title_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
                    .border_style(Style::default().fg(Color::Yellow)),
            )
            .wrap(Wrap { trim: true });

        f.render_widget(edit_paragraph, popup_area);
    }

    pub(super) fn render_entity_delete_confirm(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let entity_name = self.selected_entity.as_ref()
            .map(|e| e.canonical_name.as_str())
            .unwrap_or("?");

        let sel = self.confirm_selection;
        let sel_bg = Color::Indexed(236);
        let labels = ["Yes, delete", "No, keep it"];
        let label_width = labels.iter().map(|l| l.chars().count() + 2).max().unwrap_or(0);

        let mut lines: Vec<Line> = vec![
            Line::from(vec![
                Span::styled("Delete entity: ", Style::default().fg(Color::White)),
                Span::styled(entity_name, Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                Span::styled("?", Style::default().fg(Color::White)),
            ]),
            Line::from(""),
        ];
        for (i, label) in labels.iter().enumerate() {
            if sel == i {
                let pad = " ".repeat(label_width.saturating_sub(label.chars().count() + 2));
                lines.push(Line::from(vec![
                    Span::styled("\u{25B8} ", Style::default().fg(ACCENT).bg(sel_bg)),
                    Span::styled(format!("{}{}", label, pad), Style::default().fg(Color::White).bg(sel_bg)),
                ]));
            } else {
                lines.push(Line::from(Span::styled(format!("  {}", label), Style::default().fg(Color::DarkGray))));
            }
        }

        let max_w = lines.iter().map(|l| l.width() as u16).max().unwrap_or(0).max(30);
        let content_width = max_w.min(area.width);
        let left_pad = (area.width.saturating_sub(content_width)) / 2;
        let line_count = lines.len();
        let top_pad = (area.height as usize).saturating_sub(line_count) / 2;
        let mut centered: Vec<Line> = Vec::with_capacity(top_pad + line_count);
        for _ in 0..top_pad { centered.push(Line::from("")); }
        centered.extend(lines);

        f.render_widget(Clear, area);
        let body = Rect { x: area.x + left_pad, width: content_width, ..area };
        f.render_widget(Paragraph::new(centered).alignment(Alignment::Left), body);
    }

    pub(super) fn render_entity_merge_confirm(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let source_name = self.merge_source_entity.as_ref()
            .map(|e| e.canonical_name.as_str())
            .unwrap_or("?");
        let target_name = self.selected_entity.as_ref()
            .map(|e| e.canonical_name.as_str())
            .unwrap_or("?");

        let sel = self.confirm_selection;
        let sel_bg = Color::Indexed(236);
        let labels = ["Yes, merge", "No, cancel"];
        let label_width = labels.iter().map(|l| l.chars().count() + 2).max().unwrap_or(0);

        let mut lines: Vec<Line> = vec![
            Line::from(vec![
                Span::styled("Merge ", Style::default().fg(Color::White)),
                Span::styled(source_name, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::styled(" into ", Style::default().fg(Color::White)),
                Span::styled(target_name, Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
                Span::styled("?", Style::default().fg(Color::White)),
            ]),
            Line::from(Span::styled(
                format!("'{}' becomes an alias. Mentions transferred.", source_name),
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(""),
        ];
        for (i, label) in labels.iter().enumerate() {
            if sel == i {
                let pad = " ".repeat(label_width.saturating_sub(label.chars().count() + 2));
                lines.push(Line::from(vec![
                    Span::styled("\u{25B8} ", Style::default().fg(ACCENT).bg(sel_bg)),
                    Span::styled(format!("{}{}", label, pad), Style::default().fg(Color::White).bg(sel_bg)),
                ]));
            } else {
                lines.push(Line::from(Span::styled(format!("  {}", label), Style::default().fg(Color::DarkGray))));
            }
        }

        let max_w = lines.iter().map(|l| l.width() as u16).max().unwrap_or(0).max(30);
        let content_width = max_w.min(area.width);
        let left_pad = (area.width.saturating_sub(content_width)) / 2;
        let line_count = lines.len();
        let top_pad = (area.height as usize).saturating_sub(line_count) / 2;
        let mut centered: Vec<Line> = Vec::with_capacity(top_pad + line_count);
        for _ in 0..top_pad { centered.push(Line::from("")); }
        centered.extend(lines);

        f.render_widget(Clear, area);
        let body = Rect { x: area.x + left_pad, width: content_width, ..area };
        f.render_widget(Paragraph::new(centered).alignment(Alignment::Left).wrap(Wrap { trim: false }), body);
    }
}
