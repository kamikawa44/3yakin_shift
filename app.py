import streamlit as st
import collections
from ortools.sat.python import cp_model
import datetime
import pandas as pd
import calendar
import io

# ==============================================================================
# 3人夜勤体制用AIエンジン（最適化版）
# ==============================================================================
def create_3night_schedule(settings):
    START_DATE = settings['start_date']
    NUM_DAYS = calendar.monthrange(START_DATE.year, START_DATE.month)[1]
    NUM_NURSES = settings['num_nurses']
    staff_df = settings['staff_df']
    MONTHLY_HOLIDAYS = settings['monthly_holidays']
    
    ALL_SHIFTS = {'日勤': 0, 'ロング': 1, '準夜': 2, '深夜': 3, '休み': 4}
    SHIFT_NAMES = {v: k for k, v in ALL_SHIFTS.items()}
    ALL_SKILLS = {'師長': 0, 'リーダー': 1, '中堅': 2, '若手': 3, '新人': 4}
    ALL_BLOCKS = {name: i for i, name in enumerate(settings['block_settings']['names'])}
    
    nurse_skills = [ALL_SKILLS[staff_df.iloc[i]['スキル']] for i in range(NUM_NURSES)]
    nurse_blocks = [ALL_BLOCKS[staff_df.iloc[i]['ブロック']] for i in range(NUM_NURSES)]

    # 3人夜勤固定の設定（ロング含む）
    required_staff = {
        ALL_SHIFTS['日勤']: [settings['d_w_min'], settings['d_h_min']],
        ALL_SHIFTS['ロング']: [3, 3],  # 固定3人
        ALL_SHIFTS['準夜']: [3, 3],  # 固定3人
        ALL_SHIFTS['深夜']: [3, 3],  # 固定3人
    }
    max_day_shift_staff = { ALL_SHIFTS['日勤']: [settings['d_w_max'], settings['d_h_max']] }

    model = cp_model.CpModel()
    shifts = {}
    for n in range(NUM_NURSES):
        for d in range(NUM_DAYS):
            shifts[(n, d)] = model.NewIntVar(0, len(ALL_SHIFTS) - 1, f'shift_n{n}_d{d}')

    # --- 絶対的な制約 ---
    # 希望休の反映
    for nurse_idx, day_idx, shift_code in settings['hope_shifts']:
        if nurse_idx < NUM_NURSES and day_idx < NUM_DAYS:
            model.Add(shifts[nurse_idx, day_idx] == shift_code)
    
    # 師長の勤務（平日日勤、土日休み）
    head_nurse_indices = [i for i, skill in enumerate(nurse_skills) if skill == ALL_SKILLS['師長']]
    if head_nurse_indices:
        n = head_nurse_indices[0]
        for d in range(NUM_DAYS):
            date = START_DATE + datetime.timedelta(days=d)
            if date.weekday() < 5: 
                model.Add(shifts[n, d] == ALL_SHIFTS['日勤'])
            else: 
                model.Add(shifts[n, d] == ALL_SHIFTS['休み'])
    
    # --- 基本制約 ---
    penalties = []
    
    # 個人ごとのルール（師長以外）
    for n in range(NUM_NURSES):
        if nurse_skills[n] == ALL_SKILLS['師長']: 
            continue
        
        # ロングと準夜の数を同じにする
        is_long = []
        is_junya = []
        for d in range(NUM_DAYS):
            long_var = model.NewBoolVar(f'is_long_n{n}_d{d}')
            junya_var = model.NewBoolVar(f'is_junya_n{n}_d{d}')
            
            model.Add(shifts[n, d] == ALL_SHIFTS['ロング']).OnlyEnforceIf(long_var)
            model.Add(shifts[n, d] != ALL_SHIFTS['ロング']).OnlyEnforceIf(long_var.Not())
            model.Add(shifts[n, d] == ALL_SHIFTS['準夜']).OnlyEnforceIf(junya_var)
            model.Add(shifts[n, d] != ALL_SHIFTS['準夜']).OnlyEnforceIf(junya_var.Not())
            
            is_long.append(long_var)
            is_junya.append(junya_var)
        
        model.Add(sum(is_long) == sum(is_junya))
        
        # ロング連続禁止
        for d in range(NUM_DAYS - 1):
            model.AddBoolOr([is_long[d].Not(), is_long[d+1].Not()])
        
        # ロング→準夜パターンを推奨（ソフト制約）
        for d in range(NUM_DAYS - 1):
            long_to_junya = model.NewBoolVar(f'long_to_junya_n{n}_d{d}')
            model.AddBoolAnd([is_long[d], is_junya[d+1]]).OnlyEnforceIf(long_to_junya)
            # ペナルティを小さくして推奨程度に
            penalties.append(long_to_junya.Not() * 10)
        
        # 準夜→深夜→休みパターン（厳格）
        for d in range(NUM_DAYS - 2):
            is_shinya_next = model.NewBoolVar(f'is_shinya_n{n}_d{d+1}')
            is_yasumi_after = model.NewBoolVar(f'is_yasumi_n{n}_d{d+2}')
            
            model.Add(shifts[(n, d + 1)] == ALL_SHIFTS['深夜']).OnlyEnforceIf(is_shinya_next)
            model.Add(shifts[(n, d + 1)] != ALL_SHIFTS['深夜']).OnlyEnforceIf(is_shinya_next.Not())
            model.Add(shifts[(n, d + 2)] == ALL_SHIFTS['休み']).OnlyEnforceIf(is_yasumi_after)
            model.Add(shifts[(n, d + 2)] != ALL_SHIFTS['休み']).OnlyEnforceIf(is_yasumi_after.Not())
            
            # 準夜の次は必ず深夜
            model.AddImplication(is_junya[d], is_shinya_next)
            # 深夜の前は必ず準夜
            model.AddImplication(is_shinya_next, is_junya[d])
            # 深夜の次は必ず休み
            model.AddImplication(is_shinya_next, is_yasumi_after)
        
        # 6連勤禁止
        for d in range(NUM_DAYS - 5):
            is_work = []
            for i in range(6):
                work_var = model.NewBoolVar(f'is_work_n{n}_d{d+i}')
                model.Add(shifts[(n, d + i)] != ALL_SHIFTS['休み']).OnlyEnforceIf(work_var)
                model.Add(shifts[(n, d + i)] == ALL_SHIFTS['休み']).OnlyEnforceIf(work_var.Not())
                is_work.append(work_var)
            model.Add(sum(is_work) <= 5)

    # 各日の人数制約
    violations = {}
    for d in range(NUM_DAYS):
        date = START_DATE + datetime.timedelta(days=d)
        day_type = 0 if date.weekday() < 5 else 1
        
        for shift_code, counts in required_staff.items():
            min_required = counts[day_type]
            
            is_on_shift = []
            for n in range(NUM_NURSES):
                var = model.NewBoolVar(f'on_s{shift_code}_n{n}_d{d}')
                model.Add(shifts[(n, d)] == shift_code).OnlyEnforceIf(var)
                model.Add(shifts[(n, d)] != shift_code).OnlyEnforceIf(var.Not())
                is_on_shift.append(var)
            
            actual_count = sum(is_on_shift)
            
            if shift_code in [ALL_SHIFTS['ロング'], ALL_SHIFTS['準夜'], ALL_SHIFTS['深夜']]:
                # 夜勤は必ず3人（厳密制約）
                model.Add(actual_count == 3)
            else:
                # 日勤は柔軟に
                diff = model.NewIntVar(-NUM_NURSES, NUM_NURSES, f'diff_d{d}_s{shift_code}')
                model.Add(diff == actual_count - min_required)
                violations[(d, shift_code)] = diff
                abs_diff = model.NewIntVar(0, NUM_NURSES, f'abs_diff_d{d}_s{shift_code}')
                model.AddAbsEquality(abs_diff, diff)
                penalties.append(abs_diff * 5000)
                
                # 日勤の上限
                if shift_code == ALL_SHIFTS['日勤']:
                    max_required = max_day_shift_staff[shift_code][day_type]
                    surplus = model.NewIntVar(0, NUM_NURSES, f'surplus_d{d}_s{shift_code}')
                    model.Add(actual_count - surplus <= max_required)
                    penalties.append(surplus * 5000)
    
    # ブロック制約（3人夜勤では各ブロックから1人ずつ）
    if settings.get('enable_block_constraints', True):
        for d in range(NUM_DAYS):
            for shift_code in [ALL_SHIFTS['ロング'], ALL_SHIFTS['準夜'], ALL_SHIFTS['深夜']]:
                for block_id in ALL_BLOCKS.values():
                    nurses_in_block = [n for n, b_id in enumerate(nurse_blocks) if b_id == block_id]
                    if not nurses_in_block: 
                        continue
                    
                    block_vars = []
                    for n in nurses_in_block:
                        var = model.NewBoolVar(f'blk_n{n}_d{d}_s{shift_code}_b{block_id}')
                        model.Add(shifts[n, d] == shift_code).OnlyEnforceIf(var)
                        model.Add(shifts[n, d] != shift_code).OnlyEnforceIf(var.Not())
                        block_vars.append(var)
                    
                    # 各ブロックから1人が理想
                    model.Add(sum(block_vars) == 1)
    
    # 夜勤のリーダーシップ制約
    for d in range(NUM_DAYS):
        for shift_code in [ALL_SHIFTS['ロング'], ALL_SHIFTS['準夜'], ALL_SHIFTS['深夜']]:
            nurses_on_shift = []
            for n in range(NUM_NURSES):
                var = model.NewBoolVar(f'on_night_n{n}_d{d}_s{shift_code}')
                model.Add(shifts[(n,d)] == shift_code).OnlyEnforceIf(var)
                model.Add(shifts[(n,d)] != shift_code).OnlyEnforceIf(var.Not())
                nurses_on_shift.append((n, var))
            
            # リーダーが必ず1人以上（厳密制約）
            leader_count = sum(var for n, var in nurses_on_shift 
                            if nurse_skills[n] == ALL_SKILLS['リーダー'])
            
            model.Add(leader_count >= 1)
            
            # 新人は最大1人（ロング、準夜、深夜すべて）
            newbie_count = sum(var for n, var in nurses_on_shift 
                         if nurse_skills[n] == ALL_SKILLS['新人'])
            newbie_ok = model.NewBoolVar(f'newbie_ok_d{d}_s{shift_code}')
            model.Add(newbie_count <= 1).OnlyEnforceIf(newbie_ok)
            penalties.append(newbie_ok.Not() * 1000)
    
    # 公平性制約
    fairness_nurses = [n for n in range(NUM_NURSES) if nurse_skills[n] != ALL_SKILLS['師長']]
    
    # 休日数の公平性
    for n in fairness_nurses:
        holidays = []
        for d in range(NUM_DAYS):
            var = model.NewBoolVar(f'is_holiday_n{n}_d{d}')
            model.Add(shifts[(n, d)] == ALL_SHIFTS['休み']).OnlyEnforceIf(var)
            model.Add(shifts[(n, d)] != ALL_SHIFTS['休み']).OnlyEnforceIf(var.Not())
            holidays.append(var)
        
        num_holidays = sum(holidays)
        holiday_diff = model.NewIntVar(-NUM_DAYS, NUM_DAYS, f'holiday_diff_n{n}')
        model.Add(holiday_diff == num_holidays - MONTHLY_HOLIDAYS)
        abs_holiday_diff = model.NewIntVar(0, NUM_DAYS, f'abs_holiday_diff_n{n}')
        model.AddAbsEquality(abs_holiday_diff, holiday_diff)
        penalties.append(abs_holiday_diff * 100)
    
    # 夜勤回数の公平性
    night_counts = {}
    for n in fairness_nurses:
        night_vars = []
        for d in range(NUM_DAYS):
            var = model.NewBoolVar(f'is_night_n{n}_d{d}')
            model.Add(shifts[(n,d)] == ALL_SHIFTS['準夜']).OnlyEnforceIf(var)
            model.Add(shifts[(n,d)] != ALL_SHIFTS['準夜']).OnlyEnforceIf(var.Not())
            night_vars.append(var)
        night_counts[n] = sum(night_vars)
    
    if len(night_counts) > 1:
        min_nights = model.NewIntVar(0, NUM_DAYS, 'min_nights')
        max_nights = model.NewIntVar(0, NUM_DAYS, 'max_nights')
        model.AddMinEquality(min_nights, list(night_counts.values()))
        model.AddMaxEquality(max_nights, list(night_counts.values()))
        penalties.append((max_nights - min_nights) * 50)
    
    model.Minimize(sum(penalties))
    
    # ソルバー設定
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = settings.get('max_solve_time', 30)
    solver.parameters.num_search_workers = 4
    
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        records = []
        for n in range(NUM_NURSES):
            for d in range(NUM_DAYS):
                date = START_DATE + datetime.timedelta(days=d)
                records.append({ 
                    "スタッフ": staff_df.iloc[n]['名前'], 
                    "スキル": staff_df.iloc[n]['スキル'], 
                    "ブロック": staff_df.iloc[n]['ブロック'], 
                    "日付": date, 
                    "勤務": SHIFT_NAMES[solver.Value(shifts[(n,d)])] 
                })
        result_df = pd.DataFrame(records)
        pivot_df = result_df.pivot_table(
            index=['スタッフ','スキル','ブロック'], 
            columns='日付', 
            values='勤務', 
            aggfunc='first'
        ).reset_index()
        
        violation_report = {}
        for (d, sc), diff_var in violations.items():
            if diff_var is not None:
                diff_val = solver.Value(diff_var)
                if diff_val != 0:
                    violation_report[(d, sc)] = diff_val
                    
        return True, pivot_df, solver.ObjectiveValue(), violation_report
    else:
        return False, None, -1, None

# 集計機能
def add_summary_to_shift_table(result_df, start_date):
    """シフト表に縦集計と横集計を追加"""
    if result_df is None or result_df.empty:
        return result_df
    
    summary_df = result_df.copy()
    date_columns = [col for col in summary_df.columns if isinstance(col, (datetime.date, pd.Timestamp))]
    non_date_columns = ['スタッフ', 'スキル', 'ブロック']
    
    # 横集計（各スタッフの勤務日数）
    for idx, row in summary_df.iterrows():
        counts = {'日勤': 0, 'ロング': 0, '準夜': 0, '深夜': 0, '休み': 0}
        for col in date_columns:
            if row[col] in counts:
                counts[row[col]] += 1
        
        summary_df.loc[idx, '日勤数'] = counts['日勤']
        summary_df.loc[idx, 'ロング数'] = counts['ロング']
        summary_df.loc[idx, '準夜数'] = counts['準夜']
        summary_df.loc[idx, '深夜数'] = counts['深夜']
        summary_df.loc[idx, '休み数'] = counts['休み']
        summary_df.loc[idx, '勤務日数'] = counts['日勤'] + counts['ロング'] + counts['準夜'] + counts['深夜']
    
    # 縦集計（各日の勤務人数）
    daily_counts = {}
    for col in date_columns:
        counts = {'日勤': 0, 'ロング': 0, '準夜': 0, '深夜': 0, '休み': 0}
        for idx, row in summary_df.iterrows():
            if row[col] in counts:
                counts[row[col]] += 1
        daily_counts[col] = counts
    
    # 集計行を作成
    summary_rows = []
    for shift_type in ['日勤', 'ロング', '準夜', '深夜', '休み']:
        row_data = {
            'スタッフ': f'【{shift_type}人数】',
            'スキル': '',
            'ブロック': ''
        }
        for col in date_columns:
            row_data[col] = daily_counts[col][shift_type]
        
        row_data['日勤数'] = sum(daily_counts[col]['日勤'] for col in date_columns) if shift_type == '日勤' else ''
        row_data['ロング数'] = sum(daily_counts[col]['ロング'] for col in date_columns) if shift_type == 'ロング' else ''
        row_data['準夜数'] = sum(daily_counts[col]['準夜'] for col in date_columns) if shift_type == '準夜' else ''
        row_data['深夜数'] = sum(daily_counts[col]['深夜'] for col in date_columns) if shift_type == '深夜' else ''
        row_data['休み数'] = sum(daily_counts[col]['休み'] for col in date_columns) if shift_type == '休み' else ''
        row_data['勤務日数'] = ''
        
        summary_rows.append(row_data)
    
    # 合計勤務人数行
    total_row = {
        'スタッフ': '【合計勤務人数】',
        'スキル': '',
        'ブロック': ''
    }
    for col in date_columns:
        total_row[col] = daily_counts[col]['日勤'] + daily_counts[col]['ロング'] + daily_counts[col]['準夜'] + daily_counts[col]['深夜']
    total_row['日勤数'] = ''
    total_row['ロング数'] = ''
    total_row['準夜数'] = ''
    total_row['深夜数'] = ''
    total_row['休み数'] = ''
    total_row['勤務日数'] = sum(summary_df['勤務日数'])
    summary_rows.append(total_row)
    
    summary_rows_df = pd.DataFrame(summary_rows)
    final_df = pd.concat([summary_df, summary_rows_df], ignore_index=True)
    
    column_order = non_date_columns + date_columns + ['日勤数', 'ロング数', '準夜数', '深夜数', '休み数', '勤務日数']
    final_df = final_df[column_order]
    
    return final_df

# ==============================================================================
# UI（画面）の定義
# ==============================================================================
st.set_page_config(layout="wide")
st.title('🏥 看護師シフト管理アプリ【3人夜勤体制版】')

# セッションステートの初期化
if 'staff_df' not in st.session_state:
    initial_staff_data = {
        '名前': [f'看護師{i+1:02d}' for i in range(22)], 
        'スキル': ['師長'] + ['リーダー']*4 + ['中堅']*8 + ['若手']*6 + ['新人']*3,
        'ブロック': ['A']*7 + ['B']*7 + ['C']*8
    }
    st.session_state.staff_df = pd.DataFrame(initial_staff_data)
if 'result_df' not in st.session_state: st.session_state.result_df = None
if 'hope_shifts' not in st.session_state: st.session_state.hope_shifts = []
if 'hope_shifts_map' not in st.session_state: st.session_state.hope_shifts_map = {}
if 'violation_report' not in st.session_state: st.session_state.violation_report = {}

ALL_SHIFTS_DICT = {'日勤': 0, 'ロング': 1, '準夜': 2, '深夜': 3, '休み': 4}

tab1, tab2, tab3, tab4 = st.tabs(["シフト表", "スタッフ管理", "ルール設定", "夜勤一覧"])

with tab1:
    main_col, side_col = st.columns([4, 1])
    with side_col:
        st.header('操作')
        # デフォルトを翌月に設定
        today = datetime.date.today()
        if today.month == 12:
            default_date = datetime.date(today.year + 1, 1, 1)
        else:
            default_date = datetime.date(today.year, today.month + 1, 1)
        
        start_date_ui = st.date_input("シフト作成開始日", value=default_date, key="start_date_selector", format="YYYY/MM/DD")
        
        st.subheader('高速化設定')
        max_time = st.slider('最大計算時間（秒）', min_value=10, max_value=60, value=20, step=5)
        enable_blocks = st.checkbox('ブロック制約を有効化', value=True, 
                                   help='無効にすると計算が速くなりますが、ブロックの均等配分が保証されません')
        
        if st.button('シフト自動作成！', type="primary", use_container_width=True):
            staff_df = st.session_state.staff_df
            settings = {
                'num_nurses': len(staff_df), 
                'start_date': start_date_ui,
                'block_settings': {'names': ['A', 'B', 'C'] if 'block_names' not in st.session_state else st.session_state.block_names, 'count': len(['A', 'B', 'C'] if 'block_names' not in st.session_state else st.session_state.block_names)},
                'hope_shifts': st.session_state.hope_shifts, 
                'staff_df': staff_df,
                'monthly_holidays': st.session_state.monthly_holidays_rule,
                'd_w_min': st.session_state.d_w_min_rule, 
                'd_w_max': st.session_state.d_w_max_rule, 
                'd_h_min': st.session_state.d_h_min_rule, 
                'd_h_max': st.session_state.d_h_max_rule, 
                'max_solve_time': max_time,
                'enable_block_constraints': enable_blocks
            }
            
            with st.spinner('AIが最適なシフトを作成中です...'):
                success, result_df, objective_value, violation_report = create_3night_schedule(settings)
            
            if success:
                st.session_state.result_df = result_df
                st.session_state.violation_report = violation_report
                st.success('シフト作成に成功しました！')
                if objective_value > 0:
                    st.warning(f'一部のルールを妥協しました (ペナルティ: {int(objective_value)})。')
            else:
                st.error('時間内に解を見つけられませんでした。制約を緩和するか、計算時間を延長してください。')
        
        st.write("---")
        st.subheader("凡例")
        st.markdown("""
        <div style='margin-bottom: 10px;'>
            <div style='background-color: #4caf50; color: white; padding: 5px; margin: 2px 0; border-radius: 3px; font-weight: bold;'>
                ✅ 希望通りの勤務
            </div>
            <div style='background-color: #ff4081; color: white; padding: 5px; margin: 2px 0; border-radius: 3px; border: 3px solid #c51162; font-weight: bold;'>
                ⚠️ 希望と異なる勤務
            </div>
            <div style='background-color: #ffc107; color: #212121; padding: 5px; margin: 2px 0; border-radius: 3px; font-weight: bold;'>
                📅 土日
            </div>
            <div style='background-color: #3f51b5; color: white; padding: 5px; margin: 2px 0; border-radius: 3px; font-weight: bold;'>
                🌙 ロング・準夜・深夜 固定3人
            </div>
        </div>
        """, unsafe_allow_html=True)
                
    with main_col:
        st.header('シフト表プレビュー')
        if st.session_state.result_df is not None:
            display_df = add_summary_to_shift_table(st.session_state.result_df, start_date_ui)
            
            # 希望休の情報を整理（スタッフ名と日付と希望勤務の組み合わせ）
            hope_cells = set()
            unfulfilled_hopes = {}  # 叶わなかった希望
            for name, hopes in st.session_state.hope_shifts_map.items():
                for hope_date, hope_shift in hopes:
                    hope_cells.add((name, hope_date))
                    # 実際の勤務と希望を比較
                    if st.session_state.result_df is not None:
                        try:
                            actual_shift = st.session_state.result_df[
                                st.session_state.result_df['スタッフ'] == name
                            ][hope_date].values[0]
                            if actual_shift != hope_shift:
                                unfulfilled_hopes[(name, hope_date)] = {
                                    'hoped': hope_shift,
                                    'actual': actual_shift
                                }
                        except:
                            pass
            
            def style_shift_table(val, row_name, col_name):
                # 集計行のスタイル
                if isinstance(row_name, str) and row_name.startswith('【'):
                    if row_name == '【合計勤務人数】':
                        return 'background-color: #1a237e; color: white; font-weight: bold; opacity: 1'
                    else:
                        return 'background-color: #3949ab; color: white; opacity: 1'
                
                # 集計列のスタイル
                if col_name in ['日勤数', 'ロング数', '準夜数', '深夜数', '休み数', '勤務日数']:
                    if col_name == '勤務日数':
                        return 'background-color: #e8eaf6; color: #1a237e; font-weight: bold; opacity: 1'
                    else:
                        return 'background-color: #f5f5f5; color: #424242; opacity: 1'
                
                # 希望が叶わなかった場合の表示（濃いピンクとはっきりした枠）
                if isinstance(col_name, (datetime.date, pd.Timestamp)) and (row_name, col_name) in unfulfilled_hopes:
                    return 'background-color: #ff4081; color: white; font-weight: bold; border: 3px solid #c51162; opacity: 1'
                
                # 希望通りになった場合の表示（濃い緑）
                if isinstance(col_name, (datetime.date, pd.Timestamp)) and (row_name, col_name) in hope_cells and (row_name, col_name) not in unfulfilled_hopes:
                    return 'background-color: #4caf50; color: white; opacity: 1'
                
                # 土日のハイライト（濃い黄色）
                if isinstance(col_name, (datetime.date, pd.Timestamp)):
                    if col_name.weekday() >= 5:
                        return 'background-color: #ffc107; color: #212121; opacity: 1'
                
                return ''
            
            styled_df = display_df.style.apply(
                lambda row: [
                    style_shift_table(val, row['スタッフ'], col) 
                    for col, val in row.items()
                ], axis=1
            ).format(precision=0, na_rep='')
            
            st.dataframe(styled_df, use_container_width=True, height=700)
            
            # 希望反映状況のサマリー
            if hope_cells or unfulfilled_hopes:
                st.write("---")
                st.subheader("📊 希望反映状況")
                
                col1, col2, col3 = st.columns(3)
                
                total_hopes = len(hope_cells)
                fulfilled_hopes = len(hope_cells) - len(unfulfilled_hopes)
                
                with col1:
                    st.metric("希望総数", f"{total_hopes}件")
                
                with col2:
                    st.metric("希望通り", f"{fulfilled_hopes}件", 
                             f"{(fulfilled_hopes/total_hopes*100):.1f}%")
                
                with col3:
                    st.metric("希望と異なる", f"{len(unfulfilled_hopes)}件",
                             f"-{(len(unfulfilled_hopes)/total_hopes*100):.1f}%")
                
                # 叶わなかった希望の詳細
                if unfulfilled_hopes:
                    with st.expander("⚠️ 希望と異なる勤務の詳細"):
                        for (name, date), info in sorted(unfulfilled_hopes.items()):
                            st.write(f"• **{name}** - {date.strftime('%m/%d')}: "
                                   f"希望「{info['hoped']}」→ 実際「{info['actual']}」")
            
            # エクスポート機能
            col1, col2 = st.columns(2)
            with col1:
                csv = display_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📊 集計付きCSVダウンロード",
                    data=csv,
                    file_name=f'shift_3night_{start_date_ui.strftime("%Y%m")}.csv',
                    mime='text/csv',
                )
        else:
            st.info('サイドバーの「シフト自動作成！」ボタンを押してください。')

with tab2:
    st.header('スタッフ情報管理')
    st.info("💡 3人夜勤体制の推奨人数: 20-25名")
    
    # エクセルファイルアップロード機能
    st.subheader("📁 エクセルファイルからスタッフ情報を読み込む")
    
    # サンプルファイルのダウンロード
    with st.expander("📥 サンプルエクセルファイルをダウンロード"):
        sample_data = pd.DataFrame({
            '名前': ['師長', '田中太郎', '佐藤花子', '鈴木一郎', '高橋美咲', '渡辺健太', '伊藤愛子', '山田次郎'],
            'スキル': ['師長', 'リーダー', 'リーダー', '中堅', '中堅', '若手', '若手', '新人'],
            'ブロック': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B']
        })
        
        # エクセルファイルを作成
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            sample_data.to_excel(writer, sheet_name='スタッフ情報', index=False)
        output.seek(0)
        
        st.download_button(
            label="📥 サンプルファイルをダウンロード",
            data=output,
            file_name="スタッフ情報サンプル.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.info("""
        **ファイル形式の説明：**
        - 列1: 名前（スタッフの名前）
        - 列2: スキル（師長/リーダー/中堅/若手/新人）
        - 列3: ブロック（A/B/C などのチーム名）
        
        ※師長は必ず1名のみ、スキルを「師長」にしてください
        """)
    
    # ファイルアップロード
    uploaded_file = st.file_uploader("エクセルファイルを選択してください", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # エクセルファイルを読み込む
            df = pd.read_excel(uploaded_file)
            
            # 必要な列があるかチェック
            required_columns = ['名前', 'スキル', 'ブロック']
            if all(col in df.columns for col in required_columns):
                # データの検証
                valid_skills = ['師長', 'リーダー', '中堅', '若手', '新人']
                invalid_skills = df[~df['スキル'].isin(valid_skills)]
                
                if not invalid_skills.empty:
                    st.error(f"無効なスキルが含まれています: {invalid_skills['スキル'].unique()}")
                    st.info(f"有効なスキル: {', '.join(valid_skills)}")
                else:
                    # ブロックの取得
                    unique_blocks = sorted(df['ブロック'].unique())
                    
                    # データのプレビュー
                    st.success(f"✅ {len(df)}名のスタッフ情報を読み込みました")
                    st.write("**データプレビュー:**")
                    st.dataframe(df)
                    
                    # スキル構成の表示
                    skill_counts = df['スキル'].value_counts()
                    block_counts = df['ブロック'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**スキル構成:**")
                        for skill, count in skill_counts.items():
                            st.write(f"- {skill}: {count}名")
                    
                    with col2:
                        st.write("**ブロック構成:**")
                        for block, count in block_counts.items():
                            st.write(f"- {block}: {count}名")
                    
                    # データを適用
                    if st.button("このデータを使用する", type="primary"):
                        st.session_state.staff_df = df[required_columns].copy()
                        # ブロック設定も更新
                        st.session_state.block_names = unique_blocks
                        st.success("スタッフ情報を更新しました！")
                        st.rerun()
            else:
                st.error(f"必要な列が見つかりません。必要な列: {', '.join(required_columns)}")
                st.info("エクセルファイルに「名前」「スキル」「ブロック」の列が含まれているか確認してください。")
                
        except Exception as e:
            st.error(f"ファイルの読み込みに失敗しました: {str(e)}")
    
    st.write("---")
    st.subheader("スタッフ名簿の編集")
    edited_df = st.data_editor(
        st.session_state.staff_df, 
        column_config={
            "スキル": st.column_config.SelectboxColumn(
                "スキル", 
                options=['師長', 'リーダー', '中堅', '若手', '新人'], 
                required=True
            ), 
            "ブロック": st.column_config.SelectboxColumn(
                "ブロック", 
                options=['A', 'B', 'C'] if 'block_names' not in st.session_state else st.session_state.block_names, 
                required=True
            )
        }, 
        num_rows="dynamic", 
        use_container_width=True, 
        key="staff_editor"
    )
    st.session_state.staff_df = edited_df
    
    st.write("---")
    st.subheader("希望休の登録")
    hope_start_date = st.session_state.get('start_date_selector', datetime.date.today())
    
    # 日本語のロケール設定を追加
    import locale
    try:
        locale.setlocale(locale.LC_TIME, 'ja_JP.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_TIME, 'Japanese_Japan.932')
        except:
            pass
    
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    with col1:
        if not st.session_state.staff_df.empty:
            nurse_name = st.selectbox("対象スタッフ", options=st.session_state.staff_df['名前'])
    with col2:
        hope_date = st.date_input("希望日", value=hope_start_date, format="YYYY/MM/DD")
    with col3:
        hope_shift_name = st.selectbox("希望勤務", options=['休み', '日勤', 'ロング', '準夜', '深夜'])
    with col4:
        st.write(""); st.write("")
        if st.button("希望を追加", use_container_width=True):
            if nurse_name:
                nurse_index = st.session_state.staff_df[st.session_state.staff_df['名前'] == nurse_name].index[0]
                day_index = (hope_date - hope_start_date).days
                shift_code = ALL_SHIFTS_DICT[hope_shift_name]
                new_hope = (int(nurse_index), int(day_index), shift_code)
                if 'hope_shifts_map' not in st.session_state: 
                    st.session_state.hope_shifts_map = {}
                if nurse_name not in st.session_state.hope_shifts_map:
                    st.session_state.hope_shifts_map[nurse_name] = []
                hope_tuple = (hope_date, hope_shift_name)
                if hope_tuple not in st.session_state.hope_shifts_map[nurse_name]:
                    st.session_state.hope_shifts_map[nurse_name].append(hope_tuple)
                    st.session_state.hope_shifts.append(new_hope)
                    st.success(f"{nurse_name}の希望を追加しました。")
                    st.rerun()
                else:
                    st.warning("この希望は既に追加されています。")
                    
    st.write("---")
    st.subheader("登録済みの希望リスト")
    if not st.session_state.hope_shifts_map:
        st.info("現在、希望休の登録はありません。")
    else:
        hopes_to_delete = []
        for name, hopes in list(st.session_state.hope_shifts_map.items()):
            for i, (hope_date, hope_shift) in enumerate(hopes):
                col1, col2, col3, col4 = st.columns([2,2,2,1])
                col1.write(f"**{name}**")
                col2.write(hope_date.strftime('%Y/%m/%d'))
                col3.write(hope_shift)
                if col4.button("削除", key=f"delete_{name}_{i}"):
                    hopes_to_delete.append((name, (hope_date, hope_shift)))
        if hopes_to_delete:
            for name, hope_tuple in hopes_to_delete:
                st.session_state.hope_shifts_map[name].remove(hope_tuple)
                if not st.session_state.hope_shifts_map[name]:
                    del st.session_state.hope_shifts_map[name]
                day_index = (hope_tuple[0] - hope_start_date).days
                shift_code = ALL_SHIFTS_DICT[hope_tuple[1]]
                nurse_index = st.session_state.staff_df[st.session_state.staff_df['名前'] == name].index[0]
                hope_to_remove = (int(nurse_index), day_index, shift_code)
                if hope_to_remove in st.session_state.hope_shifts:
                    st.session_state.hope_shifts.remove(hope_to_remove)
            st.rerun()

with tab3:
    st.header('基本ルール設定【3人夜勤体制】')
    st.info('🌙 ロング・準夜・深夜は固定3人で運用されます')
    
    st.number_input('スタッフ1人あたりの公休数', min_value=8, max_value=12, value=10, key='monthly_holidays_rule')
    
    st.subheader('日勤の必要人数設定')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**平日**')
        st.number_input('日勤 (下限)', min_value=5, max_value=15, value=8, key='d_w_min_rule')
        st.number_input('日勤 (上限)', min_value=5, max_value=15, value=10, key='d_w_max_rule')
    with col2:
        st.markdown('**土日祝**')
        st.number_input('日勤 (下限)', min_value=3, max_value=10, value=5, key='d_h_min_rule')
        st.number_input('日勤 (上限)', min_value=3, max_value=10, value=7, key='d_h_max_rule')
        
    st.write("---")
    
    # 3人夜勤の特徴説明
    with st.expander("💡 3人夜勤体制の特徴"):
        st.markdown("""
        ### 夜勤の構成
        - **ロング**: 3人固定（日勤の延長）
        - **準夜**: 3人固定（16:30-1:00）
        - **深夜**: 3人固定（0:30-9:00）
        - 各ブロック（A/B/C）から1人ずつ配置
        
        ### 勤務パターン
        - ロングと準夜は同じ回数
        - ロング→準夜のパターンを推奨（連続させる）
        - 準夜→深夜→休みの3連パターン
        - ロング連続禁止
        - 6連勤禁止
        
        ### リーダーシップ要件（ロング・準夜・深夜共通）
        - **リーダーが必ず1人以上**（必須）
        - 新人は最大1人まで
        
        ### 推奨スタッフ構成（22名の場合）
        - 師長: 1名
        - リーダー: 4-6名（夜勤体制を考慮）
        - 中堅: 6-8名
        - 若手: 5-6名
        - 新人: 2-3名
        """)

with tab4:
    st.header('🌙 夜勤一覧【3人夜勤体制】')
    
    if st.session_state.result_df is not None:
        result_df = st.session_state.result_df
        date_columns = [col for col in result_df.columns if isinstance(col, (datetime.date, pd.Timestamp))]
        
        # 夜勤情報を収集
        night_shift_data = []
        
        for date_col in date_columns:
            long_members = []
            junya_members = []
            shinya_members = []
            
            for idx, row in result_df.iterrows():
                staff_name = row['スタッフ']
                skill = row['スキル']
                block = row['ブロック']
                shift = row[date_col]
                
                if shift == 'ロング':
                    long_members.append({
                        'name': staff_name,
                        'skill': skill,
                        'block': block
                    })
                elif shift == '準夜':
                    junya_members.append({
                        'name': staff_name,
                        'skill': skill,
                        'block': block
                    })
                elif shift == '深夜':
                    shinya_members.append({
                        'name': staff_name,
                        'skill': skill,
                        'block': block
                    })
            
            if long_members or junya_members or shinya_members:
                night_shift_data.append({
                    'date': date_col,
                    'long': long_members,
                    'junya': junya_members,
                    'shinya': shinya_members
                })
        
        # カレンダー形式で表示
        st.subheader('📅 月間夜勤カレンダー（3人体制）')
        
        start_date = st.session_state.get('start_date_selector', datetime.date.today())
        first_day = start_date.replace(day=1)
        calendar_start = first_day - datetime.timedelta(days=first_day.weekday())
        
        for week in range(6):
            week_start = calendar_start + datetime.timedelta(weeks=week)
            
            week_has_current_month = False
            for day in range(7):
                check_date = week_start + datetime.timedelta(days=day)
                if check_date.month == start_date.month:
                    week_has_current_month = True
                    break
            
            if not week_has_current_month and week > 0:
                break
            
            cols = st.columns(7)
            
            for day_idx in range(7):
                current_date = week_start + datetime.timedelta(days=day_idx)
                
                with cols[day_idx]:
                    day_name = ['月', '火', '水', '木', '金', '土', '日'][day_idx]
                    
                    if current_date.month != start_date.month:
                        st.markdown(f"<div style='opacity: 0.3; text-align: center;'><b>{current_date.day}日({day_name})</b></div>", unsafe_allow_html=True)
                    else:
                        if day_idx >= 5:
                            st.markdown(f"<div style='background-color: #ffc107; color: #212121; padding: 5px; text-align: center; border-radius: 5px; font-weight: bold;'><b>{current_date.day}日({day_name})</b></div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='background-color: #e3f2fd; color: #0d47a1; padding: 5px; text-align: center; border-radius: 5px; font-weight: bold;'><b>{current_date.day}日({day_name})</b></div>", unsafe_allow_html=True)
                        
                        day_data = None
                        for data in night_shift_data:
                            if data['date'] == current_date:
                                day_data = data
                                break
                        
                        if day_data:
                            # ロング（必ず3人）
                            st.markdown("**ロング** 🌅 (3人)")
                            if day_data['long']:
                                # ブロックごとにグループ化
                                blocks = {'A': [], 'B': [], 'C': []}
                                for member in day_data['long']:
                                    blocks[member['block']].append(member)
                                
                                for block, members in blocks.items():
                                    if members:
                                        for member in members:
                                            skill_color = {
                                                '師長': '#d32f2f',      # 濃い赤
                                                'リーダー': '#e91e63',  # ピンク
                                                '中堅': '#388e3c',      # 濃い緑
                                                '若手': '#1976d2',      # 濃い青
                                                '新人': '#7b1fa2'       # 濃い紫
                                            }.get(member['skill'], '#616161')
                                            
                                            st.markdown(f"<div style='background-color: {skill_color}; color: white; padding: 3px 8px; margin: 2px 0; border-radius: 3px; font-size: 11px;'>{member['block']}: {member['name']} [{member['skill']}]</div>", unsafe_allow_html=True)
                            
                            # 準夜（必ず3人）
                            st.markdown("**準夜** 🌆 (3人)")
                            if day_data['junya']:
                                # ブロックごとにグループ化
                                blocks = {'A': [], 'B': [], 'C': []}
                                for member in day_data['junya']:
                                    blocks[member['block']].append(member)
                                
                                for block, members in blocks.items():
                                    if members:
                                        for member in members:
                                            skill_color = {
                                                '師長': '#8B0000',
                                                'リーダー': '#FF6B6B',
                                                '中堅': '#4CAF50',
                                                '若手': '#42A5F5',
                                                '新人': '#BA68C8'
                                            }.get(member['skill'], '#999')
                                            
                                            st.markdown(f"<div style='background-color: {skill_color}; color: white; padding: 3px 8px; margin: 2px 0; border-radius: 3px; font-size: 11px;'>{member['block']}: {member['name']} [{member['skill']}]</div>", unsafe_allow_html=True)
                            
                            # 深夜（必ず3人）
                            st.markdown("**深夜** 🌛 (3人)")
                            if day_data['shinya']:
                                blocks = {'A': [], 'B': [], 'C': []}
                                for member in day_data['shinya']:
                                    blocks[member['block']].append(member)
                                
                                for block, members in blocks.items():
                                    if members:
                                        for member in members:
                                            skill_color = {
                                                '師長': '#8B0000',
                                                'リーダー': '#FF6B6B',
                                                '中堅': '#4CAF50',
                                                '若手': '#42A5F5',
                                                '新人': '#BA68C8'
                                            }.get(member['skill'], '#999')
                                            
                                            st.markdown(f"<div style='background-color: {skill_color}; color: white; padding: 3px 8px; margin: 2px 0; border-radius: 3px; font-size: 11px;'>{member['block']}: {member['name']} [{member['skill']}]</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='color: #DDD; text-align: center; padding: 20px;'>-</div>", unsafe_allow_html=True)
            
            st.write("---")
        
        # スキルレベルの凡例
        st.subheader('📊 スキルレベル凡例')
        skill_cols = st.columns(5)
        skills = [('師長', '#d32f2f'), ('リーダー', '#e91e63'), ('中堅', '#388e3c'), 
                 ('若手', '#1976d2'), ('新人', '#7b1fa2')]
        
        for idx, (skill, color) in enumerate(skills):
            with skill_cols[idx]:
                st.markdown(f"<div style='background-color: {color}; color: white; padding: 5px; text-align: center; border-radius: 3px; font-weight: bold;'>{skill}</div>", unsafe_allow_html=True)
        
        # 夜勤統計
        st.write("---")
        st.subheader('📈 夜勤統計（3人体制）')
        
        night_count = {}
        for idx, row in result_df.iterrows():
            staff_name = row['スタッフ']
            long_count = sum(1 for col in date_columns if row[col] == 'ロング')
            junya_count = sum(1 for col in date_columns if row[col] == '準夜')
            shinya_count = sum(1 for col in date_columns if row[col] == '深夜')
            
            if long_count > 0 or junya_count > 0 or shinya_count > 0:
                night_count[staff_name] = {
                    'skill': row['スキル'],
                    'block': row['ブロック'],
                    'long': long_count,
                    'junya': junya_count,
                    'shinya': shinya_count,
                    'total': long_count + junya_count + shinya_count
                }
        
        stats_df = pd.DataFrame([
            {
                'スタッフ': name,
                'スキル': data['skill'],
                'ブロック': data['block'],
                'ロング回数': data['long'],
                '準夜回数': data['junya'],
                '深夜回数': data['shinya'],
                '夜勤合計': data['total']
            }
            for name, data in night_count.items()
        ])
        
        if not stats_df.empty:
            stats_df = stats_df.sort_values(['ブロック', 'スキル', 'スタッフ'])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # ブロック別統計
            st.subheader('📊 ブロック別夜勤統計')
            block_stats = stats_df.groupby('ブロック').agg({
                'ロング回数': ['sum', 'mean'],
                '準夜回数': ['sum', 'mean'],
                '深夜回数': ['sum', 'mean'],
                '夜勤合計': ['sum', 'mean']
            }).round(1)
            st.dataframe(block_stats)
            
            # バランスチェック
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("最大夜勤回数", f"{stats_df['夜勤合計'].max()}回")
                st.metric("最小夜勤回数", f"{stats_df['夜勤合計'].min()}回")
            
            with col2:
                st.metric("平均夜勤回数", f"{stats_df['夜勤合計'].mean():.1f}回")
                st.metric("夜勤回数の差", f"{stats_df['夜勤合計'].max() - stats_df['夜勤合計'].min()}回")
            
            with col3:
                # ロングと準夜のペアチェック
                pair_check = all(row['ロング回数'] == row['準夜回数'] for _, row in stats_df.iterrows())
                st.metric("ロング・準夜ペア", "✅ 完全一致" if pair_check else "⚠️ 不一致あり")
                st.metric("総夜勤日数", f"{sum(stats_df['準夜回数'])}日")
    else:
        st.info('シフト表を作成してから、夜勤一覧を確認してください。')